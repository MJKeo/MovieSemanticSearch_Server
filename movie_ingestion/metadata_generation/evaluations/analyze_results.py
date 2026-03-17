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

from movie_ingestion.metadata_generation.evaluations.plot_events import SCORE_COLUMNS
from movie_ingestion.metadata_generation.evaluations.shared import (
    EVAL_DB_PATH,
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
      1. Mean/median scores per evaluation dimension (from plot_events_evaluations)
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
    # 1. Score summary (mean/median per dimension + overall_mean)
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
    )
    if scores.empty:
        print("No evaluation results found for the requested candidates.")
        conn.close()
        return None

    # ------------------------------------------------------------------
    # 2. Mean token usage per candidate
    # ------------------------------------------------------------------
    where_clause = ""
    params: list = []
    if candidate_ids:
        placeholders = ", ".join("?" * len(candidate_ids))
        where_clause = f"WHERE candidate_id IN ({placeholders})"
        params = list(candidate_ids)

    token_rows = conn.execute(
        f"""
        SELECT
            candidate_id,
            AVG(input_tokens)  AS mean_input_tokens,
            AVG(output_tokens) AS mean_output_tokens
        FROM plot_events_candidate_outputs
        {where_clause}
        GROUP BY candidate_id
        """,
        params,
    ).fetchall()

    tokens_df = pd.DataFrame(
        [dict(row) for row in token_rows],
    ).set_index("candidate_id") if token_rows else pd.DataFrame()

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

    conn.close()

    # ------------------------------------------------------------------
    # 4. Merge everything and compute cost
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
    # 5. Print formatted table
    # ------------------------------------------------------------------
    _print_plot_events_table(merged)

    return merged


def _print_plot_events_table(df: pd.DataFrame) -> None:
    """Print a formatted console table for the plot_events analysis."""
    dims = ["groundedness", "plot_summary", "character_quality", "setting"]
    short = ["grounded", "plot_summ", "char_qual", "setting"]

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

    # Score table — sorted by overall_mean descending
    scores_sorted = df.sort_values("overall_mean", ascending=False)

    header = f"{'candidate_id':<{cid_w}}"
    for s in short:
        header += f"  {s + '_mean':>{num_w}}"
        header += f"  {s + '_med':>{num_w}}"
    header += f"  {'overall':>{num_w}}"
    print(header)
    print("-" * len(header))

    for cid, row in scores_sorted.iterrows():
        line = f"{cid:<{cid_w}}"
        for d in dims:
            mean_val = row.get(f"{d}_mean", float("nan"))
            med_val = row.get(f"{d}_median", float("nan"))
            line += f"  {mean_val:>{num_w}.2f}"
            line += f"  {med_val:>{num_w}.2f}"
        overall = row.get("overall_mean", float("nan"))
        line += f"  {overall:>{num_w}.2f}"
        print(line)

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

    print("=" * 120)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analyze_plot_events()
