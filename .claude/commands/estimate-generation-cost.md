# /estimate-generation-cost

Estimate the cost of running a metadata generation type across the
full corpus, using token usage from evaluation data and movie counts
from the tracker DB.

**Usage:** `/estimate-generation-cost <generation_type>`

Example: `/estimate-generation-cost reception`

---

## The metadata type is: $ARGUMENTS

---

## Step 1: Gather token usage from evaluation data

Read all `{type}_*.json` files (not `*_evaluation.json`) in
`movie_ingestion/metadata_generation/evaluation_data/`.

For each candidate, collect:
- input_tokens and output_tokens per movie
- cost_usd per movie (if present)
- The bucket this movie belongs to (cross-reference with
  `ingestion_data/{type}_eval_buckets.json`)

Compute per-bucket averages for each candidate:
- avg input tokens, avg output tokens, avg cost per movie

---

## Step 2: Get corpus movie counts per bucket

Query the tracker DB at `ingestion_data/tracker.db` to count how
many movies fall into each bucket's input-richness range. The bucket
ranges are defined in `ingestion_data/{type}_eval_buckets.json`.

The specific query depends on the metadata type:
- **reception**: bucket by combined review character length. Query
  the tracker DB for movies at status `imdb_quality_passed` or later,
  join with IMDB data to compute combined review length, and count
  per bucket range.
- **Other types**: determine the appropriate input-richness metric
  from the generator's `build_{type}_user_prompt()` function and
  bucket accordingly.

If you can't determine the right query, ask the user what field
to bucket by.

---

## Step 3: Project costs

For each candidate, produce a table:

| Bucket | Movies | Avg Input Tok | Avg Output Tok | Avg Cost/Movie | Projected Cost |
|--------|--------|---------------|----------------|----------------|----------------|
| ultra_thin | N | ... | ... | ... | N × avg |
| ... | ... | ... | ... | ... | ... |
| **Total** | **sum** | | | | **sum** |

Show costs at both:
- **Standard API pricing** (use the model's current per-token rates)
- **Batch API pricing** (50% of standard for OpenAI models)

---

## Step 4: What-if analysis

After presenting the base projection, ask the user if they want to
model any scenarios. Common what-ifs:

- **Input truncation change**: "What if _MAX_REVIEW_CHARS changes
  from 6K to 4K?" → Re-estimate which buckets are affected and by
  how much (token reduction is roughly proportional to char reduction
  for affected buckets).
- **Output token reduction**: "What if output tokens drop 20% from
  prompt optimization?" → Apply multiplier to output token costs.
- **Model swap**: "What if we use model X instead of model Y?" →
  Apply new model's per-token pricing to the same token counts.
- **Reasoning effort change**: "What if we use minimal instead of
  low?" → Use token data from the corresponding candidate if
  available in the evaluation data.

---

## Step 5: Present summary

End with a concise comparison table of all candidates:

| Candidate | Batch Cost | Standard Cost | Avg Quality (if eval data exists) |
|-----------|-----------|---------------|-----------------------------------|

Note the cost-quality tradeoff and flag if a cheaper candidate is
within striking distance of the quality leader.
