"""
Generic analysis utilities for understanding evaluation results.

Currently supports plot_events; new metadata types can add their own
analyze_<type>() function following the same pattern.

Usage:
    python -m movie_ingestion.metadata_generation.evaluations.analyze_results

Reads from evaluation_data/eval.db. Does NOT write anything — console output only.
"""

from pathlib import Path

import pandas as pd

from movie_ingestion.metadata_generation.evaluations.plot_events import SCORE_COLUMNS, SCORE_WEIGHTS
from movie_ingestion.metadata_generation.evaluations.shared import (
    EVAL_DB_PATH,
    HIGH_SPARSITY_TMDB_IDS,
    MEDIUM_SPARSITY_TMDB_IDS,
    ORIGINAL_SET_TMDB_IDS,
    compute_score_summary,
    get_eval_connection,
)

# ---------------------------------------------------------------------------
# Pricing map — USD per 1 million tokens (input, output)
# Source: provider pricing pages as of March 2026
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_price_per_M, output_price_per_M)
    "qwen3.5-flash":                                   (0.10, 0.40),
    "gemini-2.5-flash":                                (0.30, 2.50),
    "gemini-2.5-flash-lite":                           (0.10, 0.40),
    "gpt-5-mini":                                      (0.25, 2.00),
    "gpt-5-nano":                                      (0.05, 0.40),
    "gpt-5.4-nano":                                    (0.20, 1.25),
    "openai/gpt-oss-120b":                             (0.15, 0.60),
    "meta-llama/llama-4-scout-17b-16e-instruct":       (0.11, 0.34),
}


def _compute_per_movie_cost(
    mean_input_tokens: float,
    mean_output_tokens: float,
    model: str,
) -> float | None:
    """Calculate the per-movie generation cost in USD.

    Returns None if the model is not in the pricing map.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return (mean_input_tokens * input_price + mean_output_tokens * output_price) / 1_000_000


# ---------------------------------------------------------------------------
# Plot events analysis
# ---------------------------------------------------------------------------

def analyze_plot_events(
    candidate_ids: list[str] | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame | None:
    """Print a combined quality + cost summary for plot_events candidates.

    Merges three data sources into one table:
      1. Mean scores per evaluation dimension (from plot_events_evaluations)
      2. Mean input/output token counts (from plot_events_candidate_outputs)
      3. Per-movie generation cost derived from model pricing

    Args:
        candidate_ids: If provided, restrict analysis to these candidates only.
        db_path: Override the default eval DB path (useful for testing).

    Returns:
        The merged DataFrame, or None if no evaluation data exists.
    """
    conn = get_eval_connection(db_path or EVAL_DB_PATH)

    # ------------------------------------------------------------------
    # 1. Score summary (mean per dimension + overall_mean)
    # ------------------------------------------------------------------
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='plot_events_evaluations'"
    ).fetchone()
    if not table_exists:
        print("No evaluation results found — run the evaluation pipeline first.")
        conn.close()
        return None

    scores = compute_score_summary(
        conn=conn,
        table="plot_events_evaluations",
        score_columns=SCORE_COLUMNS,
        candidate_ids=candidate_ids,
        score_weights=SCORE_WEIGHTS,
    )
    if scores.empty:
        print("No evaluation results found for the requested candidates.")
        conn.close()
        return None

    # ------------------------------------------------------------------
    # 2. Mean token usage per candidate (overall, dense-only, sparse-only)
    # ------------------------------------------------------------------
    def _query_mean_tokens(
        extra_conditions: list[str] | None = None,
        extra_params: list | None = None,
    ) -> pd.DataFrame:
        """Query mean input/output tokens from candidate_outputs with optional filters."""
        conditions: list[str] = []
        params_list: list = []
        if candidate_ids:
            placeholders = ", ".join("?" * len(candidate_ids))
            conditions.append(f"candidate_id IN ({placeholders})")
            params_list.extend(candidate_ids)
        if extra_conditions:
            conditions.extend(extra_conditions)
            params_list.extend(extra_params or [])
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = conn.execute(
            f"""
            SELECT
                candidate_id,
                AVG(input_tokens)  AS mean_input_tokens,
                AVG(output_tokens) AS mean_output_tokens
            FROM plot_events_candidate_outputs
            {where}
            GROUP BY candidate_id
            """,
            params_list,
        ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows]).set_index("candidate_id")

    # Overall token averages (used for the cost table)
    tokens_df = _query_mean_tokens()

    # Dense-only and sparse-only token averages (used for the value ranking table)
    dense_placeholders = ", ".join("?" * len(ORIGINAL_SET_TMDB_IDS))
    dense_tokens_df = _query_mean_tokens(
        extra_conditions=[f"movie_id IN ({dense_placeholders})"],
        extra_params=list(ORIGINAL_SET_TMDB_IDS),
    )

    sparse_movie_ids_list = MEDIUM_SPARSITY_TMDB_IDS + HIGH_SPARSITY_TMDB_IDS
    sparse_placeholders = ", ".join("?" * len(sparse_movie_ids_list))
    sparse_tokens_df = _query_mean_tokens(
        extra_conditions=[f"movie_id IN ({sparse_placeholders})"],
        extra_params=list(sparse_movie_ids_list),
    )

    # ------------------------------------------------------------------
    # 3. Model + provider from candidates table
    # ------------------------------------------------------------------
    candidate_rows = conn.execute(
        "SELECT candidate_id, model, provider FROM candidates "
        "WHERE metadata_type = 'plot_events'"
    ).fetchall()

    candidates_df = pd.DataFrame(
        [dict(row) for row in candidate_rows],
    ).set_index("candidate_id") if candidate_rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # 4. Score summaries for dense and sparse movie subsets
    # ------------------------------------------------------------------
    dense_scores = compute_score_summary(
        conn=conn,
        table="plot_events_evaluations",
        score_columns=SCORE_COLUMNS,
        candidate_ids=candidate_ids,
        movie_ids=ORIGINAL_SET_TMDB_IDS,
        score_weights=SCORE_WEIGHTS,
    )
    sparse_scores = compute_score_summary(
        conn=conn,
        table="plot_events_evaluations",
        score_columns=SCORE_COLUMNS,
        candidate_ids=candidate_ids,
        movie_ids=sparse_movie_ids_list,
        score_weights=SCORE_WEIGHTS,
    )

    conn.close()

    # ------------------------------------------------------------------
    # 5. Merge everything and compute cost
    # ------------------------------------------------------------------
    merged = scores.join(tokens_df, how="left").join(candidates_df, how="left")

    # Compute per-movie cost for each candidate
    costs = []
    for cid, row in merged.iterrows():
        model = row.get("model")
        mean_in = row.get("mean_input_tokens")
        mean_out = row.get("mean_output_tokens")
        if pd.notna(model) and pd.notna(mean_in) and pd.notna(mean_out):
            costs.append(_compute_per_movie_cost(mean_in, mean_out, model))
        else:
            costs.append(None)
    merged["cost_per_movie"] = costs

    # ------------------------------------------------------------------
    # 6. Print formatted tables
    # ------------------------------------------------------------------
    _print_plot_events_table(
        merged, dense_scores, sparse_scores,
        dense_tokens_df, sparse_tokens_df,
    )

    return merged


def _print_score_table(
    df: pd.DataFrame,
    title: str,
    cid_w: int,
    num_w: int = 10,
) -> None:
    """Print a score table with overall_mean first, then per-dimension means."""
    dims = ["groundedness", "plot_summary", "character_quality", "setting"]
    short = ["grounded", "plot_summ", "char_qual", "setting"]

    scores_sorted = df.sort_values("overall_mean", ascending=False)

    print()
    print(f"--- {title} ---")
    header = f"{'candidate_id':<{cid_w}}  {'overall':>{num_w}}"
    for s in short:
        header += f"  {s + '_mean':>{num_w}}"
    print(header)
    print("-" * len(header))

    for cid, row in scores_sorted.iterrows():
        overall = row.get("overall_mean", float("nan"))
        line = f"{cid:<{cid_w}}  {overall:>{num_w}.2f}"
        for d in dims:
            mean_val = row.get(f"{d}_mean", float("nan"))
            line += f"  {mean_val:>{num_w}.2f}"
        print(line)


def _print_plot_events_table(
    df: pd.DataFrame,
    dense_scores: pd.DataFrame,
    sparse_scores: pd.DataFrame,
    dense_tokens: pd.DataFrame,
    sparse_tokens: pd.DataFrame,
) -> None:
    """Print formatted console tables for the plot_events analysis."""
    # Determine column widths
    cid_w = max(12, df.index.str.len().max() + 1)
    prov_w = 10
    model_w = max(8, int(df["model"].str.len().max()) + 1) if "model" in df.columns else 8
    num_w = 10
    cost_w = 14

    # Header
    print()
    print("=" * 120)
    print("plot_events evaluation — quality scores, token usage, and per-movie cost")
    print("=" * 120)

    # All-movies score table
    _print_score_table(df, "All movies", cid_w, num_w)

    # Dense and sparse subset score tables
    if not dense_scores.empty:
        _print_score_table(dense_scores, "Dense movie performance", cid_w, num_w)
    if not sparse_scores.empty:
        _print_score_table(sparse_scores, "Sparse movie performance", cid_w, num_w)

    # Cost table — sorted by cost_per_movie ascending
    cost_sorted = df.sort_values("cost_per_movie", ascending=True, na_position="last")

    print()
    cost_header = (
        f"{'candidate_id':<{cid_w}}"
        f"  {'provider':<{prov_w}}"
        f"  {'model':<{model_w}}"
        f"  {'mean_in_tok':>{num_w}}"
        f"  {'mean_out_tok':>{num_w + 1}}"
        f"  {'cost/movie':>{cost_w}}"
    )
    print(cost_header)
    print("-" * len(cost_header))

    for cid, row in cost_sorted.iterrows():
        provider = row.get("provider", "?")
        model = row.get("model", "?")
        mean_in = row.get("mean_input_tokens", float("nan"))
        mean_out = row.get("mean_output_tokens", float("nan"))
        cost = row.get("cost_per_movie")

        # Format tokens as integers, cost as USD with 6 decimal places
        in_str = f"{mean_in:>{num_w},.0f}" if pd.notna(mean_in) else f"{'n/a':>{num_w}}"
        out_str = f"{mean_out:>{num_w + 1},.0f}" if pd.notna(mean_out) else f"{'n/a':>{num_w + 1}}"
        cost_str = f"${cost:>{cost_w - 1},.6f}" if cost is not None else f"{'n/a':>{cost_w}}"

        print(
            f"{cid:<{cid_w}}"
            f"  {provider:<{prov_w}}"
            f"  {model:<{model_w}}"
            f"  {in_str}"
            f"  {out_str}"
            f"  {cost_str}"
        )

    # Value ranking table — sorted by overall_mean * cost_per_movie ascending
    # Lower product = better value (high quality at low cost)
    _print_value_ranking_table(df, dense_tokens, sparse_tokens, cid_w, num_w, cost_w)

    print("=" * 120)
    print()


def _print_value_ranking_table(
    df: pd.DataFrame,
    dense_tokens: pd.DataFrame,
    sparse_tokens: pd.DataFrame,
    cid_w: int,
    num_w: int,
    cost_w: int,
) -> None:
    """Print a value-ranking table with separate dense/sparse cost columns.

    Computes cost/1K movies from the dense-only and sparse-only mean token
    counts so users can see the cost difference between data-rich and
    data-sparse movies.
    """
    # Need overall_mean and a model to compute costs
    required_cols = {"overall_mean", "model"}
    if not required_cols.issubset(df.columns):
        return

    value_df = df[["overall_mean", "model"]].dropna()
    if value_df.empty:
        return

    value_df = value_df.copy()

    # Compute per-movie cost for dense and sparse subsets separately
    def _cost_per_1k(tokens_df: pd.DataFrame, candidate_id: str, model: str) -> float | None:
        if candidate_id not in tokens_df.index:
            return None
        row = tokens_df.loc[candidate_id]
        mean_in = row.get("mean_input_tokens")
        mean_out = row.get("mean_output_tokens")
        if pd.isna(mean_in) or pd.isna(mean_out):
            return None
        per_movie = _compute_per_movie_cost(mean_in, mean_out, model)
        return per_movie * 1_000 if per_movie is not None else None

    dense_costs = []
    sparse_costs = []
    for cid, row in value_df.iterrows():
        model = row["model"]
        dense_costs.append(_cost_per_1k(dense_tokens, cid, model))
        sparse_costs.append(_cost_per_1k(sparse_tokens, cid, model))

    value_df["dense_cost_1k"] = dense_costs
    value_df["sparse_cost_1k"] = sparse_costs

    # Drop candidates missing both cost columns
    has_any_cost = value_df["dense_cost_1k"].notna() | value_df["sparse_cost_1k"].notna()
    value_df = value_df[has_any_cost]
    if value_df.empty:
        return

    value_sorted = value_df.sort_values("overall_mean", ascending=False)

    cost_1k_w = 16
    print()
    val_header = (
        f"{'candidate_id':<{cid_w}}"
        f"  {'overall':>{num_w}}"
        f"  {'dense cost/1K':>{cost_1k_w}}"
        f"  {'sparse cost/1K':>{cost_1k_w}}"
    )
    print(val_header)
    print("-" * len(val_header))

    for cid, row in value_sorted.iterrows():
        overall = row["overall_mean"]
        dense_c = row["dense_cost_1k"]
        sparse_c = row["sparse_cost_1k"]
        dense_str = f"${dense_c:>{cost_1k_w - 1},.2f}" if dense_c is not None else f"{'n/a':>{cost_1k_w}}"
        sparse_str = f"${sparse_c:>{cost_1k_w - 1},.2f}" if sparse_c is not None else f"{'n/a':>{cost_1k_w}}"
        print(
            f"{cid:<{cid_w}}"
            f"  {overall:>{num_w}.2f}"
            f"  {dense_str}"
            f"  {sparse_str}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analyze_plot_events()
