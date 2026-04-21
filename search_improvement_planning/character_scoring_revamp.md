# Character Scoring Revamp

Add prominence-aware scoring for the character sub-type of the step 3
entity endpoint (parallel to actor prominence scoring), and compress
all prominence scores into [0.5, 1.0] so that a match never drops
below the 0.5 floor used for dealbreaker-eligible scores.

Today, character lookups return a binary 1.0 for any match. This
loses the signal that "Spider-Man movies" is a stronger statement
than "movies with Spider-Man in them". Actors already have zone-based
prominence scoring — characters should have an analogous (but
simpler) system.

---

## Locked Decisions

1. **Two character modes, not four.** `CENTRAL` (character is the
   subject of the query) and `DEFAULT` (character is a filter with
   a gentle prominence preference). No SUPPORTING or MINOR —
   "supporting character named X" is essentially never how users
   phrase queries, and "cameo as X" is the weird case we're
   comfortable ignoring.

2. **Character `cast_size` is tracked separately from actor
   `cast_size`.** Characters are not 1:1 with actors: a single
   actor can be credited with multiple character names
   ("Peter Parker" / "Spider-Man") and one actor can play several
   characters. Store `character_cast_size` on the character
   postings table directly.

3. **`billing_position` for a character = its index in the flat
   per-edge character list produced by the parser.** The IMDB
   GraphQL parser already emits characters in cast-edge order, with
   each character name on a given edge appended sequentially (see
   `movie_ingestion/imdb_scraping/parsers.py:491-503`). So a top-
   billed actor's first character gets position 1, their second
   alias character gets position 2, the second-billed actor's first
   character gets position 3, etc. Position reflects the "credit
   cascade" the way a reader would encounter it.

4. **Drop and rebuild `lex.inv_character_postings` from scratch.**
   Raw IMDB data lives in `ingestion_data/imdb/{tmdb_id}.json` and
   in the tracker's `imdb_data` table, so no information is lost by
   dropping the posting table and re-populating from source. This
   is simpler than an in-place ALTER + backfill, and guarantees
   every row has the new columns without NULL-handling gymnastics.

5. **Scoring floor at 0.5 for all prominence scoring (actors +
   characters).** Any matched entity gets at least 0.5. No match
   still yields "no row / no score" — the floor only applies when a
   row exists. Implemented as a single affine transform at the tail
   of `_prominence_score`: `0.5 + 0.5 * raw` plus a defensive clamp.

6. **CENTRAL mode is `cast_size`-independent.** Unlike actor
   prominence (which uses sqrt-scaled zones to adapt to cast size),
   CENTRAL uses a fixed decay: position 1–2 score 1.0, then decay
   by 0.15 per position until the floor. "Spider-Man movies" has
   the same intent whether the cast is 8 or 80.

7. **DEFAULT mode for characters is a simple linear ramp** from 1.0
   at the top position to 0.5 at the last position — no zones.

8. **Entity endpoint only** — these changes are scoped to the step
   3 entity endpoint under `search_v2/stage_3/`. The legacy
   compound-lexical character code path (`_build_compound_character_cte`
   and friends in `db/postgres.py`) is not touched here; it still
   returns a binary matched count, which is the right behavior for
   its caller.

---

## Scoring Formulas

### Character CENTRAL mode

Fixed curve, ignores `character_cast_size`. Applied before the
[0.5, 1.0] compression:

```
raw(pos) = max(0.0, 1.0 - 0.15 * max(0, pos - 2))
```

| billing_position | raw  | compressed (final) |
|------------------|------|--------------------|
| 1                | 1.00 | 1.000              |
| 2                | 1.00 | 1.000              |
| 3                | 0.85 | 0.925              |
| 4                | 0.70 | 0.850              |
| 5                | 0.55 | 0.775              |
| 6                | 0.40 | 0.700              |
| 7                | 0.25 | 0.625              |
| 8+               | 0.10 → 0.00 | 0.550 → 0.500 |

### Character DEFAULT mode

Linear ramp, uses `character_cast_size`. Applied before compression:

```
raw(pos, size) = 1.0 - (pos - 1) / max(1, size - 1)
```

After compression: `0.5 + 0.5 * raw`, which ranges smoothly from
1.0 (top) to 0.5 (bottom) regardless of cast size.

### Actor modes (unchanged shapes, compressed)

Keep the four existing mode scorers (`_score_in_default_mode`,
`_score_in_lead_mode`, `_score_in_supporting_mode`,
`_score_in_minor_mode`) byte-for-byte. The only change is the
final transform in `_prominence_score`:

```python
raw = _MODE_SCORERS[mode](zone, zp)
compressed = 0.5 + 0.5 * raw
return max(0.5, min(1.0, compressed))
```

New effective ranges per mode (for reference during review):

| Mode | Lead zone | Supporting zone | Minor zone |
|------|-----------|-----------------|------------|
| DEFAULT    | 1.0          | 0.925 → 0.85  | 0.85 → 0.75  |
| LEAD       | 1.0          | 0.80 → 0.70   | 0.70 → 0.60  |
| SUPPORTING | 0.85 → 0.80  | 1.0           | 0.80 → 0.675 |
| MINOR      | 0.675 → 0.625| 0.75 → 0.675  | 0.85 → 1.0   |

Known quirk: MINOR mode now floors at 0.625 for a lead-role match
(user asked for "cameo", got a lead). Ordering within mode is still
correct (minor-zone match beats lead-zone match). Deliberately left
as-is — the 0.5 floor rule is absolute.

---

## Schema Changes

### File: `db/init/01_create_postgres_tables.sql`

Replace the current `lex.inv_character_postings` definition
(currently at lines 262–269) with:

```sql
-- Inverted index postings for character names (with billing metadata
-- for prominence scoring). Analogous to inv_actor_postings, but with
-- a distinct cast_size because characters are not 1:1 with actors
-- (aliases like "Peter Parker" + "Spider-Man" produce multiple
-- character rows for a single cast edge).
CREATE TABLE IF NOT EXISTS lex.inv_character_postings (
  term_id                   BIGINT NOT NULL,
  movie_id                  BIGINT NOT NULL,
  billing_position          INT    NOT NULL,
  character_cast_size       INT    NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_character_postings_movie
  ON lex.inv_character_postings (movie_id);
```

PK is still `(term_id, movie_id)` — if a character name appears
twice for the same movie (two different cast edges credit the same
character name), ingestion takes the first (topmost-billed)
occurrence via `dict.fromkeys`, matching the actor convention.

---

## Ingestion Changes

### File: `movie_ingestion/final_ingestion/ingest_movie.py`

Current code around lines 374–428 collects characters into a flat
list and calls `batch_insert_character_postings(term_ids, movie_id)`
with no position. Change required:

1. **Track position alongside normalization.** Replace the loop
   that builds `characters: list[str]` with one that builds
   `(normalized_character, billing_position)` tuples, where
   `billing_position` starts at 1 and increments for every
   non-empty normalized character (empty strings are skipped
   without consuming a position, same as today).

2. **Dedup preserving position.** After resolving term IDs, walk
   the ordered `(term_id, position)` list and keep only the first
   occurrence of each term_id — analogous to the actor
   `dict.fromkeys(actor_term_ids)` pattern, but carrying position.

3. **Compute `character_cast_size`.** The count of distinct
   character term_ids kept after dedup — i.e., `len(deduped)`.
   This is the same shape as the actor `cast_size` computation.

4. **Call the updated insert helper** (see next section) with
   `term_ids`, `billing_positions`, `movie_id`, `character_cast_size`.

### File: `db/postgres.py`

Replace `batch_insert_character_postings` (currently at lines
634–648) with a signature mirroring `batch_insert_actor_postings`
(lines 530–551):

```python
async def batch_insert_character_postings(
    term_ids: list[int],
    movie_id: int,
    *,
    character_cast_size: int,
    conn=None,
) -> None:
    """Insert character postings for one movie. billing_position
    is implicit from list order (1-indexed). character_cast_size is
    applied uniformly to every row for this movie."""
    # INSERT with unnest() over (term_ids, billing_positions), same
    # pattern as batch_insert_actor_postings.
```

---

## Backfill Script

New script: `movie_ingestion/final_ingestion/rebuild_character_postings.py`

Purpose: drop `lex.inv_character_postings`, recreate it with the
new schema, and re-populate from the raw IMDB data already stored
in the tracker's `imdb_data` table. Single-purpose, one-shot.

Steps:

1. Open a Postgres connection and execute:
   ```sql
   DROP TABLE IF EXISTS lex.inv_character_postings;
   ```
   followed by the new CREATE TABLE DDL (duplicate of the init
   script, kept in this file so it is self-contained).

2. Iterate movies that are in status `ingested` in the tracker
   (`movie_ingestion.tracker`). For each movie:
   - Load the IMDB JSON from `ingestion_data/imdb/{tmdb_id}.json`
     (source of truth; matches what ingestion reads), or fall back
     to the tracker's `imdb_data` row if the JSON isn't present.
   - Run the same normalization + dedup pipeline as the updated
     ingestion code to derive `(term_id, billing_position)` pairs
     and `character_cast_size`.
   - Resolve term IDs via `batch_upsert_lexical_dictionary` (safe:
     idempotent upsert) and `batch_upsert_character_strings`.
   - Call the new `batch_insert_character_postings`.

3. Commit in batches (e.g. every 500 movies) for crash safety,
   logging progress.

4. Run in the same environment as regular ingestion (same DB
   creds, same normalization helpers). Idempotent: rerunning after
   a failure picks up where it left off because the DROP + CREATE
   already happened and inserts are `ON CONFLICT DO NOTHING`
   (follow the actor posting insert's conflict handling).

Notes:
- No retention concerns — raw data is in `imdb_data` / IMDB JSON
  files, so a full rebuild is safe.
- Does not touch `lex.character_strings` or
  `lex.lexical_dictionary`; those stay populated and don't need
  rebuilding.

---

## Step 2 Prompt Changes

### File: `search_v2/stage_2.py` (around lines 199–333, entity
endpoint guidance)

Add character prominence language to the description-writing
instructions, parallel to the existing actor prominence guidance.
Extend the examples so the LLM preserves CENTRAL vs DEFAULT
signals:

- `"Spider-Man movies"` / `"Batman films"` / `"movies about the Joker"`
  → description: `"centers on the character <Name>"` (CENTRAL)
- `"movies with Spider-Man in it"` / `"films featuring the Joker"`
  → description: `"includes the character <Name>"` (DEFAULT)
- `"character named X"` with no prominence signal →
  `"includes the character X"` (DEFAULT)

The rule of thumb for the LLM: if the query uses the character
name as the *subject* of a noun phrase like "X movies" or
"films about X", write CENTRAL language; if the character is
described as appearing *in* or *featured in* the movie, write
DEFAULT language.

---

## Step 3 Entity-Generation LLM Changes

### File: `search_v2/stage_3/entity_query_generation.py`

1. **Add a `_CHARACTER_PROMINENCE` section** to the system prompt,
   parallel to `_ACTOR_PROMINENCE` (lines 196–243). Two modes:
   - `default` — no prominence language, or DEFAULT-style phrasing
     ("includes the character X", "features X"). This is the
     typical case.
   - `central` — the description frames the character as the
     subject ("centers on the character X", "X is the protagonist",
     "movies about X"). Only choose CENTRAL when the description
     contains language that pins the character to the center of
     the film.

2. **Add output fields** on `EntityQuerySpec` (see schema section
   below):
   - `character_prominence_evidence: Optional[str]`
   - `character_prominence_mode: Optional[CharacterProminenceMode]`

3. **Update the field-level instructions in the prompt** with a
   block parallel to the `prominence_evidence` /
   `actor_prominence_mode` block (lines 348–362). The evidence
   field must quote or paraphrase the language that justified the
   mode, or state "no prominence signal" explicitly when DEFAULT.

4. **Assembly scope.** These fields are only populated when
   `entity_type == EntityType.PERSON` with a character sub-type,
   or when the entity search targets `lex.character_strings`. The
   existing specific-role dispatch in step 3 execution already
   identifies character lookups — follow the same gate.

---

## Schema / Enum Additions

### File: `schemas/enums.py`

Add a new enum:

```python
class CharacterProminenceMode(str, Enum):
    DEFAULT = "default"
    CENTRAL = "central"
```

### File: `schemas/entity_translation.py`

1. **Add the two new fields** to `EntityQuerySpec`:
   ```python
   character_prominence_evidence: Optional[str] = None
   character_prominence_mode: Optional[CharacterProminenceMode] = None
   ```

2. **Update the post-parse validator** (currently lines 135–153).
   Add a character-sub-type branch parallel to the actor branch:
   - When the spec targets character lookups and
     `character_prominence_evidence is None`, coerce to
     `"no prominence signal"`.
   - When `character_prominence_mode is None`, coerce to
     `CharacterProminenceMode.DEFAULT`.

   Leave actor coercion untouched.

---

## Step 3 Execution Changes

### File: `search_v2/stage_3/entity_query_execution.py`

1. **Replace `_fetch_binary_role_scores` for characters** with a
   new `_fetch_character_scores(term_ids, mode, restrict_movie_ids)`
   that mirrors `_fetch_actor_scores`:
   - New helper in `db/postgres.py`:
     `fetch_character_billing_rows(term_ids, restrict_movie_ids)`
     returning `list[tuple[movie_id, billing_position, character_cast_size]]`.
     Direct analogue of `fetch_actor_billing_rows` (lines 354–396
     of `db/postgres.py`).
   - For each row, compute score via a new `_character_prominence_score`
     function that dispatches on `CharacterProminenceMode`:
     - `CENTRAL` → CENTRAL curve above.
     - `DEFAULT` → linear ramp above.
   - Apply the same `0.5 + 0.5 * raw` compression + clamp used for
     actors. Consider factoring a tiny helper
     `_compress_to_floor(raw: float) -> float` so both actor and
     character paths share the transform.
   - If a character term_id somehow matches two rows for one movie
     (shouldn't — PK prevents it, but the actor path has the same
     defensive max; keep it).

2. **Update `_prominence_score`** (lines 160–189) to apply the
   compression transform. Single-point change; do not edit the
   individual `_score_in_*_mode` functions.

3. **Update `_execute_character`** (lines 342–372) to:
   - Read `spec.character_prominence_mode` (already defaulted to
     `CharacterProminenceMode.DEFAULT` by the validator).
   - Call `_fetch_character_scores` instead of
     `_fetch_binary_role_scores`.
   - Leave the character-name resolution logic (lookup_text +
     `character_alternative_names`) unchanged.

4. **`_fetch_binary_role_scores` stays** for the other non-actor
   role sub-types (directors, writers, producers, composers).
   Only the character path moves off of it.

---

## Testing Notes & Edge Cases

These are called out for future test planning, not implemented
here (per the test-boundaries rule).

- **Duplicate character names across cast edges.** If "The Joker"
  is credited to two different actors in the same movie (rare
  multi-actor role), ingestion's `dict.fromkeys` keeps the
  topmost-billed occurrence, which is the correct choice for
  CENTRAL scoring.
- **Aliases on one edge.** A cast edge with `["Peter Parker",
  "Spider-Man"]` produces two character rows with sequential
  positions. Both appear in the posting table; a query for
  "Spider-Man" hits the row at its own position (typically 2 if
  Peter Parker is listed first). Under CENTRAL mode, both still
  score 1.0 (positions 1 and 2 are tied at the top).
- **Small casts.** A 2-character movie under DEFAULT scores
  position 1 → 1.0, position 2 → 0.5. Edge case where
  `character_cast_size == 1`: `max(1, size - 1)` protects the
  division; the single character scores 1.0.
- **Compression floor.** Post-change, no matched entity should
  ever score below 0.5 on any prominence path. Worth a unit test
  that exercises each mode at its minimum raw output.
- **Actor score distribution shift.** Callers downstream of
  `_fetch_actor_scores` now see a compressed range. Anywhere that
  assumed scores in [0, 1] (e.g. normalization in score merging)
  should still work since [0.5, 1.0] is a strict subset — but
  worth scanning `search_v2/stage_4/` callers for implicit
  assumptions about the minimum.

---

## Open Items

None. All design questions from the planning conversation are
resolved. Ready to stage the implementation:

1. Schema + DDL
2. Enum + `EntityQuerySpec` field additions + validator
3. Updated `batch_insert_character_postings` helper
4. Updated `ingest_movie.py` character handling
5. Backfill script (drops table, rebuilds from imdb_data)
6. Compression transform on actor `_prominence_score`
7. New `fetch_character_billing_rows` + `_fetch_character_scores`
8. Updated `_execute_character` dispatch
9. Step 2 prompt: CENTRAL vs DEFAULT description guidance
10. Step 3 entity LLM prompt: `_CHARACTER_PROMINENCE` section and
    output fields
