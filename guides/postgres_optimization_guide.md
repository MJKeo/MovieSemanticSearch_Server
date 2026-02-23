# Postgres Optimization Guide

This guide covers 6 unresolved inefficiencies in the Postgres layer of the movie search application. Each section describes the problem, why it matters, and the concrete implementation path to fix it.

All changes are scoped to three files: `postgres.py`, `lexical_search.py`, and `ingest_movie.py`.

---

## Fix 1 — Single Compound Lexical Query (Eligible CTE Deduplication)

### Problem

Every bucket search (`search_people_postings`, `search_character_postings`, `search_studio_postings`, `search_title_postings`, etc.) independently builds and sends a full SQL query to Postgres, each containing its own copy of the `eligible AS MATERIALIZED (...)` CTE. For a single user search request, Postgres receives N separate queries and re-scans `movie_card` to materialize the identical eligible set N times.

### What to Change

Replace all individual bucket search functions (`search_people_postings`, `search_studio_postings`, `search_character_postings`, `search_character_postings_by_query`, `search_title_postings`) and their underlying postgres.py functions (`fetch_phrase_postings_match_counts`, `fetch_character_match_counts`, `fetch_character_match_counts_by_query`, `fetch_title_token_match_scores`) with a single compound query builder and executor.

### New postgres.py Function

Create one function that accepts all resolved term IDs for every bucket, conditionally includes CTEs only for buckets that have data, and returns tagged rows:

```python
async def execute_compound_lexical_search(
    *,
    people_term_ids: list[int],
    studio_term_ids: list[int],
    character_query_idxs: list[int],
    character_term_ids: list[int],
    title_searches: list[TitleSearchInput],  # dataclass with token_idxs, term_ids, f_coeff, k
    filters: Optional[MetadataFilters] = None,
    exclude_movie_ids: Optional[set[int]] = None,
) -> CompoundLexicalResult:  # dataclass holding per-bucket results
```

### SQL Shape

The function builds a single SQL string:

```sql
WITH eligible AS MATERIALIZED (
    SELECT movie_id, title_token_count
    FROM public.movie_card
    WHERE release_ts >= $1 AND genre_ids && $2::int[]
),

-- Only included if people_term_ids is non-empty
people_matches AS (
    SELECT p.movie_id, COUNT(DISTINCT p.term_id)::int AS matched
    FROM lex.inv_person_postings p
    JOIN eligible e ON e.movie_id = p.movie_id
    WHERE p.term_id = ANY($3::bigint[])
      AND NOT (p.movie_id = ANY($4::bigint[]))
    GROUP BY p.movie_id
),

-- Only included if studio_term_ids is non-empty
studio_matches AS (
    ...same pattern against inv_studio_postings...
),

-- Only included if character_term_ids is non-empty.
-- Groups by query_idx so franchise scoring can use per-phrase granularity.
character_matches AS (
    SELECT qc.query_idx, p.movie_id, COUNT(DISTINCT qc.query_idx)::int AS matched
    FROM (SELECT unnest($X::int[]) AS query_idx, unnest($Y::bigint[]) AS term_id) qc
    JOIN lex.inv_character_postings p ON p.term_id = qc.term_id
    JOIN eligible e ON e.movie_id = p.movie_id
    WHERE NOT (p.movie_id = ANY($Z::bigint[]))
    GROUP BY qc.query_idx, p.movie_id
),

-- One title CTE per title search (title_search_0, title_search_1, ...)
-- Each follows the existing token_matches → title_matches → title_scored pattern
q_tokens_0 AS (
    SELECT unnest($A::int[]) AS token_idx, unnest($B::bigint[]) AS term_id
),
token_matches_0 AS (
    SELECT DISTINCT p.movie_id, qt.token_idx
    FROM q_tokens_0 qt
    JOIN lex.inv_title_token_postings p ON p.term_id = qt.term_id
    JOIN eligible e ON e.movie_id = p.movie_id
    WHERE NOT (p.movie_id = ANY($C::bigint[]))
),
title_matches_0 AS (
    SELECT movie_id, COUNT(*)::int AS m FROM token_matches_0 GROUP BY movie_id
),
title_scored_0 AS (
    SELECT tm.movie_id,
           ($f_coeff * ((tm.m::double precision / $k) * (tm.m::double precision / e.title_token_count)))
           / ($beta_sq * (tm.m::double precision / e.title_token_count) + (tm.m::double precision / $k))
           AS title_score
    FROM title_matches_0 tm
    JOIN eligible e ON e.movie_id = tm.movie_id
    WHERE e.title_token_count > 0 AND $k > 0
)

-- Final UNION ALL with bucket labels
SELECT 'people' AS bucket, -1 AS query_idx, movie_id, matched::double precision AS score FROM people_matches
UNION ALL
SELECT 'studio' AS bucket, -1 AS query_idx, movie_id, matched::double precision AS score FROM studio_matches
UNION ALL
SELECT 'character' AS bucket, query_idx, movie_id, matched::double precision AS score FROM character_matches
UNION ALL
SELECT 'title_0' AS bucket, -1 AS query_idx, movie_id, title_score AS score FROM title_scored_0 WHERE title_score >= $threshold
UNION ALL
SELECT 'title_1' AS bucket, -1 AS query_idx, movie_id, title_score AS score FROM title_scored_1 WHERE title_score >= $threshold
```

### Conditional CTE Assembly

The function should build the CTE list dynamically. Each bucket is a block of CTE text that is only appended when that bucket has data:

```python
cte_parts: list[str] = []
params: list = []

# Always include eligible when filters are active
if use_eligible:
    eligible_cte, eligible_params = _build_eligible_cte(filters)
    cte_parts.append(eligible_cte)
    params.extend(eligible_params)

# People — only if we have term IDs
if people_term_ids:
    cte_parts.append(_build_people_cte(...))
    params.extend(...)

# Studios — only if we have term IDs
if studio_term_ids:
    cte_parts.append(_build_studio_cte(...))
    params.extend(...)

# Characters — only if we have term IDs
if character_term_ids:
    cte_parts.append(_build_character_cte(...))
    params.extend(...)

# Title searches — one set of CTEs per title search
for i, ts in enumerate(title_searches):
    if ts.term_ids:
        cte_parts.append(_build_title_ctes(i, ...))
        params.extend(...)

# Build UNION ALL from whichever buckets were included
union_parts = []
if people_term_ids:
    union_parts.append("SELECT 'people' AS bucket, ...")
...
```

### Result Parsing

The single result set comes back as rows of `(bucket, query_idx, movie_id, score)`. Parse in Python:

```python
people_scores: dict[int, int] = {}
studio_scores: dict[int, int] = {}
character_by_query: dict[int, dict[int, int]] = {}
title_scores_by_search: dict[int, dict[int, float]] = {}

for bucket, query_idx, movie_id, score in rows:
    if bucket == "people":
        people_scores[movie_id] = int(score)
    elif bucket == "studio":
        studio_scores[movie_id] = int(score)
    elif bucket == "character":
        character_by_query.setdefault(query_idx, {})[movie_id] = int(score)
    elif bucket.startswith("title_"):
        search_idx = int(bucket.split("_")[1])
        title_scores_by_search.setdefault(search_idx, {})[movie_id] = score
```

### What to Remove from postgres.py

- `fetch_phrase_postings_match_counts`
- `fetch_character_match_counts`
- `fetch_character_match_counts_by_query`
- `fetch_title_token_match_scores`

These are all absorbed into the compound query builder. `_build_eligible_cte` stays — it's used as a building block by the new function.

### What to Remove from lexical_search.py

- `search_people_postings`
- `search_studio_postings`
- `search_character_postings`
- `search_character_postings_by_query`
- `search_title_postings`

The `asyncio.gather` in STEP 6 that dispatches these is replaced by a single call to the compound function.

### Handling the No-Filters Case

When no metadata filters are active, the `eligible` CTE is omitted. Posting CTEs that previously did `JOIN eligible e ON e.movie_id = p.movie_id` instead reference `public.movie_card` directly, but only in CTEs that need `title_token_count` (i.e., title scoring). Phrase bucket CTEs (people, studio, character) don't need the join at all when there's no eligible filter — they just query the posting table directly.

Build this conditionality into each CTE builder helper:

```python
def _build_people_cte(use_eligible: bool, ...) -> str:
    join = "JOIN eligible e ON e.movie_id = p.movie_id" if use_eligible else ""
    return f"""people_matches AS (
        SELECT p.movie_id, COUNT(DISTINCT p.term_id)::int AS matched
        FROM lex.inv_person_postings p
        {join}
        WHERE p.term_id = ANY(%s::bigint[])
        GROUP BY p.movie_id
    )"""
```

---

## Fix 2 — Batch Dictionary Upserts During Ingestion

### Problem

In `ingest_lexical_data`, every string gets its own individual round-trip to Postgres via `upsert_lexical_dictionary`. A typical movie generates ~31 sequential INSERT ... ON CONFLICT ... RETURNING calls (5 title tokens + 15 people + 8 characters + 3 studios). Each acquires a connection, runs a transaction, commits, and releases. At 150K movies this totals ~4.65 million individual writes just for dictionary entries.

On top of that, `upsert_title_token_string` and `upsert_character_string` each add another per-item call, pushing the total to ~44 round-trips per movie.

### What to Change

Replace all per-string `upsert_lexical_dictionary` calls with a single batch upsert that processes all strings for one movie at once. Similarly batch `upsert_title_token_string` and `upsert_character_string`.

### New postgres.py Functions

**Batch dictionary upsert:**

```python
async def batch_upsert_lexical_dictionary(
    norm_strings: list[str],
    conn=None,
) -> dict[str, int]:
    """
    Batch upsert normalized strings into lex.lexical_dictionary.

    Returns mapping of norm_str → string_id for every input string.
    Uses a single round-trip regardless of list size.
    """
    if not norm_strings:
        return {}

    # Deduplicate while preserving order
    unique_strings = list(dict.fromkeys(norm_strings))

    query = """
        WITH input_strings AS (
            SELECT unnest(%s::text[]) AS norm_str
        ),
        inserted AS (
            INSERT INTO lex.lexical_dictionary (norm_str, created_at)
            SELECT norm_str, now()
            FROM input_strings
            ON CONFLICT (norm_str) DO NOTHING
            RETURNING norm_str, string_id
        )
        SELECT norm_str, string_id FROM inserted
        UNION ALL
        SELECT d.norm_str, d.string_id
        FROM lex.lexical_dictionary d
        JOIN input_strings i ON i.norm_str = d.norm_str
        WHERE NOT EXISTS (
            SELECT 1 FROM inserted ins WHERE ins.norm_str = d.norm_str
        )
    """
    # Execute on shared connection if provided, otherwise acquire one
    rows = await _execute_on_conn(conn, query, (unique_strings,))
    return {row[0]: row[1] for row in rows}
```

Note: the CTE pattern ensures we get `string_id` back for both newly inserted and already-existing rows in a single query, without the write amplification of `ON CONFLICT DO UPDATE SET touched_at = now()`.

**Batch title_token_strings upsert:**

```python
async def batch_upsert_title_token_strings(
    string_ids: list[int],
    norm_strings: list[str],
    conn=None,
) -> None:
    """Batch upsert title token lookup rows."""
    if not string_ids:
        return

    query = """
        INSERT INTO lex.title_token_strings (string_id, norm_str)
        SELECT unnest(%s::bigint[]), unnest(%s::text[])
        ON CONFLICT (string_id) DO UPDATE SET
            norm_str = EXCLUDED.norm_str
    """
    await _execute_on_conn(conn, query, (string_ids, norm_strings))
```

**Batch character_strings upsert (same pattern):**

```python
async def batch_upsert_character_strings(
    string_ids: list[int],
    norm_strings: list[str],
    conn=None,
) -> None:
    if not string_ids:
        return

    query = """
        INSERT INTO lex.character_strings (string_id, norm_str)
        SELECT unnest(%s::bigint[]), unnest(%s::text[])
        ON CONFLICT (string_id) DO UPDATE SET
            norm_str = EXCLUDED.norm_str
    """
    await _execute_on_conn(conn, query, (string_ids, norm_strings))
```

### What to Remove from postgres.py

- `upsert_lexical_dictionary` (individual version)
- `upsert_title_token_string` (individual version)
- `upsert_character_string` (individual version)
- `upsert_phrase_term` (was a wrapper around `upsert_lexical_dictionary`)

### Changes to ingest_movie.py

`ingest_lexical_data` collects all strings first, then does one batch upsert:

```python
async def ingest_lexical_data(movie: BaseMovie, conn=None) -> None:
    movie_id = int(getattr(movie, "tmdb_id"))

    # Phase 1: Collect all strings that need dictionary IDs
    title_tokens = [normalize_string(t) for t in movie.normalized_title_tokens()]
    title_tokens = [t for t in title_tokens if t]

    people = list(create_people_list(movie))  # already normalized

    raw_characters = getattr(movie, "characters", []) or []
    characters = [normalize_string(c) for c in raw_characters if normalize_string(c)]

    raw_studios = getattr(movie, "production_companies", []) or []
    studios = [normalize_string(s) for s in raw_studios if normalize_string(s)]

    # Phase 2: Single batch dictionary upsert for ALL strings
    all_strings = list(dict.fromkeys(title_tokens + people + characters + studios))
    string_id_map = await batch_upsert_lexical_dictionary(all_strings, conn=conn)

    # Phase 3: Resolve IDs and batch-insert into sub-tables
    title_token_ids = [string_id_map[t] for t in title_tokens if t in string_id_map]
    people_ids = [string_id_map[p] for p in people if p in string_id_map]
    character_ids = [string_id_map[c] for c in characters if c in string_id_map]
    studio_ids = [string_id_map[s] for s in studios if s in string_id_map]

    # Phase 4: Parallel batch inserts (all use the shared connection)
    await asyncio.gather(
        batch_upsert_title_token_strings(
            title_token_ids,
            [t for t in title_tokens if t in string_id_map],
            conn=conn,
        ),
        batch_upsert_character_strings(
            character_ids,
            [c for c in characters if c in string_id_map],
            conn=conn,
        ),
        batch_insert_title_token_postings(list(dict.fromkeys(title_token_ids)), movie_id, conn=conn),
        batch_insert_person_postings(list(dict.fromkeys(people_ids)), movie_id, conn=conn),
        batch_insert_character_postings(list(dict.fromkeys(character_ids)), movie_id, conn=conn),
        batch_insert_studio_postings(list(dict.fromkeys(studio_ids)), movie_id, conn=conn),
    )
```

This reduces ~44 round-trips per movie down to ~7 (1 dictionary batch + 6 parallel sub-table inserts), and further down to ~2 if the sub-table inserts share a connection (see Fix 3).

---

## Fix 3 — Shared Connection Per Movie Ingestion

### Problem

Each call to `_execute_write` acquires its own connection from the pool, executes one statement, commits, and releases. During `ingest_movie`, dozens of writes cycle through the pool sequentially. With `max_size=10`, this causes unnecessary pool contention during burst ingestion and prevents transactional atomicity — a crash mid-ingest can leave the movie_card written but postings incomplete.

### What to Change

Add a connection-passing pattern to all write functions so that a single movie's entire ingestion runs on one connection with one transaction.

### New postgres.py Helper

```python
async def _execute_on_conn(
    conn,
    query: str,
    params: Sequence[object] | None = None,
    fetch: bool = False,
):
    """
    Execute a query on an existing connection, or acquire one from the pool.

    When conn is provided, executes without committing (caller manages transaction).
    When conn is None, falls back to the existing _execute_write behavior.
    """
    if conn is not None:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchall() if fetch else None
    else:
        # Fallback: standalone connection (backward-compatible)
        async with pool.connection() as fallback_conn:
            async with fallback_conn.cursor() as cur:
                await cur.execute(query, params)
                result = await cur.fetchall() if fetch else None
            await fallback_conn.commit()
            return result
```

### Changes to All Write Functions

Every batch insert and upsert function gets an optional `conn=None` parameter. When provided, it uses `_execute_on_conn` instead of `_execute_write`. The function signature for `batch_insert_title_token_postings` becomes:

```python
async def batch_insert_title_token_postings(
    term_ids: list[int],
    movie_id: int,
    conn=None,
) -> None:
    if not term_ids:
        return
    query = """
        INSERT INTO lex.inv_title_token_postings (term_id, movie_id)
        SELECT unnest(%s::bigint[]), %s
        ON CONFLICT (term_id, movie_id) DO NOTHING
    """
    await _execute_on_conn(conn, query, (term_ids, movie_id))
```

Apply the same pattern to: `batch_insert_person_postings`, `batch_insert_character_postings`, `batch_insert_studio_postings`, `batch_upsert_title_token_strings`, `batch_upsert_character_strings`, `upsert_movie_card`, and `batch_upsert_lexical_dictionary`.

### Changes to ingest_movie.py

Wrap the entire movie ingestion in a single connection and transaction:

```python
async def ingest_movie(movie: BaseMovie) -> None:
    """Ingest one BaseMovie into movie_card plus all lexical posting tables."""
    async with pool.connection() as conn:
        try:
            await asyncio.gather(
                ingest_movie_card(movie, conn=conn),
                ingest_lexical_data(movie, conn=conn),
            )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
```

This gives you:
- **1 connection per movie** instead of ~44
- **1 commit per movie** instead of ~44
- **Atomicity** — if any step fails, the entire movie's data is rolled back
- **No pool contention** — even bulk-ingesting many movies won't exhaust the pool because each movie holds only one connection

### Important: asyncio.gather with a Shared Connection

Note that `asyncio.gather` with a shared psycopg `AsyncConnection` is safe as long as you don't have two coroutines issuing queries on the same connection simultaneously. Psycopg's `AsyncConnection` uses pipeline mode internally to handle this, but if you encounter issues, convert the `gather` in Phase 4 of `ingest_lexical_data` to sequential awaits. At that point you're already down to ~7 queries per movie, so sequential execution on one connection is still far better than 44 separate connections.

---

## Fix 4 — Remove `touched_at` from lexical_dictionary

### Problem

`upsert_lexical_dictionary` currently does `ON CONFLICT (norm_str) DO UPDATE SET touched_at = now()`. This forces a full row UPDATE (WAL write, dead tuple, autovacuum work) every time an existing string is re-ingested, which happens constantly for common terms. The `touched_at` column is not read anywhere in the query path and has no planned use.

### What to Change

**Step 1 — Schema migration:**

```sql
ALTER TABLE lex.lexical_dictionary DROP COLUMN touched_at;
```

This is a metadata-only operation in Postgres. It doesn't rewrite the table — existing rows keep the bytes on disk but the column becomes invisible. Fast and safe.

**Step 2 — Update the batch upsert (from Fix 2):**

The batch upsert already uses `ON CONFLICT DO NOTHING`, so no code change is needed beyond ensuring the old `upsert_lexical_dictionary` function (which referenced `touched_at`) is fully removed.

If you want to confirm, the new batch upsert pattern:

```sql
INSERT INTO lex.lexical_dictionary (norm_str, created_at)
SELECT unnest(%s::text[]) AS norm_str, now()
ON CONFLICT (norm_str) DO NOTHING
RETURNING norm_str, string_id
```

No mention of `touched_at`. Existing rows are untouched — zero write amplification.

**Step 3 — Remove from `upsert_movie_card` if referenced:**

Check if `touched_at` appears in the movie_card upsert or anywhere else. It should only exist on `lexical_dictionary`, but verify. `movie_card` has its own `updated_at` column which serves a different purpose (tracking when movie metadata last changed) and should be kept.

---

## Fix 5 — Consolidate Character Term Resolution

### Problem

Character term resolution (`_resolve_character_term_ids`) hits `lex.character_strings` with LIKE + trigram scans. It is called 3 separate times during a single search:

1. **STEP 4**: for exclude characters + exclude franchise phrases
2. **STEP 6** (inside `search_character_postings`): for include characters
3. **STEP 6** (inside `search_character_postings_by_query`): for include franchise phrases

Each call is an independent DB round-trip with a LIKE scan, the most expensive of the resolution operations.

### What to Change

Resolve all character phrases — include, franchise, and exclude — in a single batch during STEP 4.

### Changes to lexical_search.py

**STEP 4 — Combine all character phrase resolution:**

```python
# Combine all character phrases into one resolution call
all_character_phrases = _dedupe_preserve_order(
    include_characters + include_franchise_phrases + exclude_characters + exclude_franchise_phrases
)

# Build an index map so we can split results back after resolution
include_char_range = range(0, len(include_characters))
franchise_char_offset = len(include_characters)
franchise_char_range = range(franchise_char_offset, franchise_char_offset + len(include_franchise_phrases))
exclude_char_offset = franchise_char_offset + len(include_franchise_phrases)
# exclude chars occupy the rest

(
    phrase_id_map,
    all_title_token_maps,
    all_character_term_map,   # Single resolution for ALL character phrases
    exclude_title_term_ids,
) = await asyncio.gather(
    fetch_phrase_term_ids(all_exact_phrases),
    _resolve_all_title_tokens(all_title_searches),
    _resolve_character_term_ids(all_character_phrases),
    _resolve_exact_exclude_title_term_ids(all_exclude_title_tokens),
)

# Split the resolved term map back into per-group maps
include_character_term_map = {
    idx: all_character_term_map[idx]
    for idx in include_char_range
    if idx in all_character_term_map
}
franchise_character_term_map = {
    idx - franchise_char_offset: all_character_term_map[idx]
    for idx in franchise_char_range
    if idx in all_character_term_map
}
exclude_character_term_ids = list({
    tid
    for idx in range(exclude_char_offset, len(all_character_phrases))
    if idx in all_character_term_map
    for tid in all_character_term_map[idx]
})
```

**STEP 6 — Pass pre-resolved term IDs directly:**

With Fix 1 (compound query), character term IDs are passed directly into the compound query builder. The builder receives `(query_idxs, term_ids)` arrays — it never needs to call `_resolve_character_term_ids` internally.

If you implement this before Fix 1, you'll need to modify `search_character_postings` and `search_character_postings_by_query` to accept pre-resolved term maps instead of raw phrase strings:

```python
async def search_character_postings(
    character_term_map: dict[int, list[int]],  # pre-resolved, not raw phrases
    filters: Optional[MetadataFilters] = None,
    exclude_movie_ids: Optional[set[int]] = None,
) -> dict[int, int]:
    if not character_term_map:
        return {}
    # Flatten and go straight to posting join — no resolution step
    query_idxs, term_ids = [], []
    for q_idx, tids in character_term_map.items():
        for tid in tids:
            query_idxs.append(q_idx)
            term_ids.append(tid)
    ...
```

### Net Effect

3 LIKE-scan DB round-trips → 1. This is the most expensive resolution type because LIKE with trigram indexing requires scanning the GIN index and confirming pattern matches, whereas exact phrase lookups use a simple B-tree equality check.

---

## Fix 6 — Consolidate Duplicate Character Match Functions

### Problem

`fetch_character_match_counts` and `fetch_character_match_counts_by_query` in postgres.py are structurally near-identical. They build the same CTE chain, same eligible join, same exclusion clause. The only difference is whether the final SELECT groups by `movie_id` alone or by `(query_idx, movie_id)`.

Similarly, `search_character_postings` and `search_character_postings_by_query` in lexical_search.py are parallel wrappers around these functions with identical resolution logic.

### What to Change

If you implement Fix 1 (compound query), both functions are deleted entirely — character matching becomes a single CTE in the compound query that always groups by `(query_idx, movie_id)`. The Python-side parsing splits or aggregates as needed.

If you implement this independently of Fix 1, keep only the `_by_query` variant (which is the superset) and add an aggregation helper:

### Changes to postgres.py

Delete `fetch_character_match_counts`. Keep only `fetch_character_match_counts_by_query` and rename it to `fetch_character_match_counts`:

```python
async def fetch_character_match_counts(
    query_idxs: list[int],
    term_ids: list[int],
    use_eligible: bool,
    filters: Optional[MetadataFilters] = None,
    exclude_movie_ids: Optional[set[int]] = None,
) -> dict[int, dict[int, int]]:
    """
    Per-query-index character match counts from inv_character_postings.

    Returns dict keyed by query_idx, each containing {movie_id: matched_count}.
    Callers that need flat {movie_id: total_matched} use aggregate_match_counts().
    """
    # ... existing _by_query implementation unchanged ...
```

### Changes to lexical_search.py

Delete `search_character_postings`. Keep only `search_character_postings_by_query`, rename it to `search_character_postings`, and add the aggregation helper:

```python
def aggregate_character_results(
    by_query: list[dict[int, int]],
) -> dict[int, int]:
    """
    Collapse per-query character match maps into {movie_id: total_matched_phrases}.

    Used for regular include_characters where all matches contribute equally
    to the aggregated character score.
    """
    totals: dict[int, int] = {}
    for query_map in by_query:
        for movie_id, matched in query_map.items():
            totals[movie_id] = totals.get(movie_id, 0) + matched
    return totals
```

In `lexical_search()`, the include_characters call becomes:

```python
include_character_by_query = search_character_postings(
    include_character_term_map, filters, excluded_movie_ids
)
character_scores = aggregate_character_results(include_character_by_query)
```

And franchise characters use the per-query results directly as they already do.

---

## Implementation Order

These fixes have dependencies on each other. Recommended order:

| Order | Fix | Reason |
|-------|-----|--------|
| 1 | Fix 4 (remove touched_at) | Schema-only, zero risk, immediate WAL reduction |
| 2 | Fix 6 (consolidate character functions) | Pure refactor, reduces code before bigger changes |
| 3 | Fix 5 (batch character resolution) | Reduces STEP 4 DB calls, independent of query structure |
| 4 | Fix 2 (batch dictionary upserts) | Ingestion improvement, independent of search path |
| 5 | Fix 3 (shared connection) | Builds on Fix 2, requires conn parameter threading |
| 6 | Fix 1 (compound query) | Largest change, benefits from Fixes 5+6 being done first |

Fix 1 is the highest-impact change for search latency but also the largest code change. Fixes 2-3 are the highest-impact for ingestion throughput. Fixes 4-6 are lower-risk and can be done first to reduce code complexity before tackling Fix 1.