# /evaluate-metadata-results

Analyze evaluation data for a metadata generation type and produce a
structured report.

**Usage:** `/evaluate-metadata-results <generation_type>`

Example: `/evaluate-metadata-results reception`

---

## The metadata type is: $ARGUMENTS

---

## Step 1: Gather data

1. Read the system prompt from
   `movie_ingestion/metadata_generation/prompts/{type}.py`
2. Read the output schema from
   `movie_ingestion/metadata_generation/schemas.py` — find the
   `{Type}Output` class
3. Read the bucket definitions from
   `ingestion_data/{type}_eval_buckets.json` — note the bucket names,
   ranges, and which movies are in each bucket
4. Use an Explore agent to read ALL `{type}_*.json` files (not
   `*_evaluation.json`) in
   `movie_ingestion/metadata_generation/evaluation_data/`. For each
   file, extract: tmdb_id, title, candidate names, each candidate's
   full result object, input_tokens, output_tokens, cost_usd.
5. Use a second Explore agent (in parallel) to read ALL
   `{type}_*_evaluation.json` files in the same directory. For each
   file, extract: tmdb_id, candidate names, per-axis grades, and
   justification text.

If no evaluation JSON files exist for the given type, stop and report
"No evaluation data found for {type}".

---

## Step 2: Build the report

Produce a structured report with the following sections. Use the
actual data — do not summarize generically.

### 2a. Overview
- Number of movies evaluated
- Number of candidates
- Candidate names (model + config)
- Evaluation axes used
- Bucket breakdown: how many movies per bucket

### 2b. Aggregate scores table
For each candidate, show:
- Average score per evaluation axis
- Overall average across all axes
- Total cost across all evaluated movies

Sort candidates by overall average (descending).

### 2c. Per-bucket analysis
For each bucket (ordered thin → rich):
- Average overall score per candidate in that bucket
- Note any candidate that is notably stronger or weaker in this
  bucket vs its overall average
- Flag any systematic patterns (e.g., "candidate X struggles with
  thin-input movies")

### 2d. Per-axis deep dive
For each evaluation axis:
- Which candidate is strongest/weakest on this axis?
- Are there movies where ALL candidates scored low on this axis?
  If so, list them — these may indicate prompt or schema issues
  rather than model issues.

### 2e. Per-movie failure analysis
List any movies where ANY candidate scored 1 or 2 on ANY axis.
For each:
- Movie title and tmdb_id
- Which candidate(s) failed and on which axis
- The justification text for the low score
- The bucket this movie belongs to

Group failures by failure pattern if common themes emerge (e.g.,
"over-extraction on thin input", "hallucinated content",
"generic topic-listing").

### 2f. Cross-candidate comparison
For movies where candidates diverge significantly (spread of 2+
points on any axis):
- List the movie, the axis, and each candidate's score
- Quote the justification text to explain the divergence
- Note what the stronger candidate did differently

### 2g. Recommendations
Based on the data:
- Which candidate appears strongest overall?
- Are there specific buckets or axes where a different candidate
  would be preferred?
- What prompt or schema changes might address systematic failures?
- Any movies that should be added/removed from the eval set?

---

## Step 3: Present the report

Present the full report to the user. After presenting, ask if they
want to dive deeper into any section or specific movies.
