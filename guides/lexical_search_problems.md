# Lexical Search — Bugs & Fixes

---

## 1. Eligible CTE Omits `title_token_count`

`_build_eligible_cte()` in `postgres.py` (line 289) selects only `movie_id` from `public.movie_card`. The title scoring CTE compensates by re-joining `public.movie_card` later (line 994) to get `title_token_count`. This works but forces a redundant join against the full table when filters are active. The guide (§8) specifies the eligible CTE should carry `title_token_count` directly. Additionally, the guide references `lex.movies` as the eligible-set source while the code uses `public.movie_card`.

**Fix:** Add `title_token_count` to the eligible CTE's SELECT list. Update the `title_scored` CTE to read from `eligible` instead of re-joining `movie_card`.

---

## 2. `include_franchise_title_searches` Silently Rebuilt from Wrong Source

During STEP 1 (lines 345–358), `include_franchise_title_searches` is built by tokenizing the raw `source` string. Then in STEP 2 (lines 377–379), the list is **completely replaced**:

```python
include_franchise_title_searches = [
    tokenize_title_phrase(phrase) for phrase in include_franchise_phrases
]
```

This re-tokenizes from `include_franchise_phrases`, which contains `normalize_string(source)` outputs — not the original `source`. If `normalize_string` and the internal normalization within `tokenize_title_phrase` ever diverge in behavior, the token lists will silently differ. The rebuild also obscures intent: it's doing dedup-by-reconstruction rather than explicit dedup.

**Fix:** Don't build `include_franchise_title_searches` during STEP 1 at all. Derive it once from the deduped `include_franchise_phrases` in STEP 2, making the single source of truth explicit.

---

## 3. `params` Type Mismatch — Lists Passed Where Tuples Expected

`_execute_read` is typed as `params: tuple | None`, but every dynamic query builder (`fetch_phrase_postings_match_counts`, `fetch_title_token_match_scores`, `fetch_character_match_counts`) passes a `list`. psycopg3 accepts this at runtime, but it violates the type contract and would be flagged by any static type checker.

**Fix:** Either call `tuple(params)` before passing to `_execute_read`, or widen the type hint to `Sequence`.

---

## 4. Exclude Title Tokens Have No `max_df` Filtering

The guide (§9.7) specifies that exclude title tokens should use "exact lookup + max_df filtering." The code resolves exclude title tokens via `fetch_phrase_term_ids` (lines 408–411, 432–434), which does exact dictionary lookup — but applies **no** `doc_frequency` check. If an exclude token like "the" has df > 10,000, the anti-join will exclude nearly every movie with "the" in its title.

The include side correctly filters by max_df through `fetch_title_token_ids`, but the exclude side bypasses this entirely because it uses the phrase dictionary path instead of the title token path.

**Fix:** After resolving exclude title token term_ids, filter them against `lex.title_token_doc_frequency` to discard any with `doc_frequency > MAX_DF` before passing to posting queries. Alternatively, resolve exclude title tokens through `fetch_title_token_ids` (exact-only tier) which already enforces max_df.

---

## 5. Hardcoded `cte_parts[0]` in `fetch_phrase_postings_match_counts`

Line 695 of `postgres.py`:

```python
with_clause = f"WITH {cte_parts[0]}\n        " if cte_parts else ""
```

This assumes `cte_parts` will never contain more than one element. Currently safe (only the eligible CTE can be appended), but fragile. The other two dynamic query builders (`fetch_title_token_match_scores`, `fetch_character_match_counts`) both use `",\n        ".join(cte_parts)`.

**Fix:** Use the join pattern for consistency:
```python
with_clause = f"WITH {', '.join(cte_parts)}" if cte_parts else ""
```

---

## 6. Layered Division-by-Zero Guards in Title Scoring SQL

The title scoring query guards against division by zero with `WHERE mc.title_token_count > 0 AND %s > 0` (where `%s` binds to `k`). But `k = len(token_term_id_map)`, which counts token *positions*, not resolved term_ids. A map like `{0: [], 1: []}` would pass the `if not token_term_id_map` check at line 261 with `k = 2`, but produce an empty `term_ids` list — caught only by the second check at line 272. The guards work in practice but rely on multiple sequential checks rather than a single clear invariant.

**Fix:** Add an explicit early return after flattening:
```python
if not term_ids:
    return {}
```
This already exists at line 272, so this is technically fine. For clarity, assert the invariant: `k` should equal the number of *non-empty* token positions, not total positions. Consider computing `k` from the filtered map rather than the raw map.

---

## 7. SQL Table Name Interpolation in `fetch_phrase_postings_match_counts`

The `table` parameter is f-string-interpolated into SQL at lines 699 and 683. An allowlist check at line 659 mitigates injection risk, but the pattern of raw string interpolation for table names is a maintenance hazard.

**Fix:** Replace the string parameter with an enum or a frozen mapping:
```python
class PostingTable(Enum):
    PERSON = "lex.inv_person_postings"
    STUDIO = "lex.inv_studio_postings"
```
Then accept `PostingTable` instead of `str`, eliminating the runtime allowlist check entirely.

---

## 8. Franchise Character Search Fires One Query Per Phrase

Lines 450–453:
```python
franchise_character_futures = [
    search_character_postings([phrase], filters, exclude_character_term_ids)
    for phrase in include_franchise_phrases
]
```

Each franchise phrase gets its own `search_character_postings` call (substring resolution + posting join). With 3 franchises, that's 3 separate DB round-trips.

The per-franchise split exists because scoring takes `max(title_score, character_score)` per franchise. However, `search_character_postings` already returns per-`query_idx` match counts via the `q_chars` CTE. A single batched call with all franchise phrases would return the same per-phrase granularity.

**Fix:** Call `search_character_postings(include_franchise_phrases, filters, exclude_character_term_ids)` once. The returned `{movie_id: matched_count}` would then need to be decomposed per-phrase. This requires the function to return per-query-idx detail (or a new variant that does), but the SQL already tracks `query_idx`.

---

## 9. Single-Row Ingestion Inserts

`insert_title_token_posting`, `insert_person_posting`, `insert_character_posting`, and `insert_studio_posting` (lines 378–439) each insert one row per call, each acquiring its own connection. Ingesting a movie with 20 cast members = 20 separate round-trips.

The guide (§5.5) specifies batch insertion:
```sql
INSERT INTO lex.inv_person_postings (term_id, movie_id)
SELECT unnest($1::bigint[]), $2
ON CONFLICT DO NOTHING;
```

**Fix:** Replace single-row insert functions with batch variants:
```python
async def batch_insert_person_postings(term_ids: list[int], movie_id: int) -> None:
    query = """
        INSERT INTO lex.inv_person_postings (term_id, movie_id)
        SELECT unnest(%s::bigint[]), %s
        ON CONFLICT DO NOTHING;
    """
    await _execute_write(query, (term_ids, movie_id))
```
Repeat for the other three posting tables. Keep the single-row variants only if there's a legitimate one-at-a-time use case.

---

## 10. `raise e` Destroys Original Traceback

Lines 708–709 and 1019–1020:
```python
except Exception as e:
    raise e
```

`raise e` resets the traceback to the current frame, losing the original stack trace from inside the Postgres driver. Since the TODO logging isn't implemented, these try/except blocks currently do nothing but harm debuggability.

**Fix:** Use bare `raise`, or remove the try/except entirely until logging is implemented:
```python
# Either:
except Exception:
    raise

# Or just remove the try/except wrapper
search_results = await _execute_read(query, params)
return {row[0]: float(row[1]) for row in search_results}
```