# Franchise Metadata Generation

LLM-generated franchise membership data for each movie. Identifies whether a
movie belongs to a recognizable intellectual property or brand, classifies its
role within that franchise, and captures culturally established grouping names.

Replaces the current title-token + character-matching franchise heuristic in
lexical search with a dedicated `franchise_membership` table and
`lex.inv_franchise_postings` posting table.

---

## Franchise Definition

**Any recognizable intellectual property or brand that originated in any
medium** — film series, video games, toys, books, comics, TV shows, board
games, theme parks, etc. — where the movie is an adaptation, extension, or
product of that IP.

This matches the definition already used by the lexical entity extractor in
`implementation/prompts/lexical_prompts.py` ("a specific multi-film media
brand or series name"), broadened to explicitly include non-film-originated IPs.

Examples:
- **Film series:** "Star Wars", "Fast and Furious", "Mission Impossible"
- **Video game IPs:** "Super Mario", "Sonic the Hedgehog", "Resident Evil",
  "Tomb Raider", "Mortal Kombat", "Angry Birds"
- **Toy/product IPs:** "Barbie", "Transformers", "G.I. Joe", "Hot Wheels", "LEGO"
- **Book/comic IPs:** "Harry Potter", "Lord of the Rings", "Dune",
  "Marvel Cinematic Universe"
- **TV-originated IPs:** "Scooby-Doo", "SpongeBob SquarePants",
  "The Simpsons"

**The franchise name is the IP name, not the film series name.** "Mario" or
"Super Mario" — not "Super Mario Bros. Movie series". This aligns with how
users search ("Mario movie", not "Super Mario Bros. Movie franchise").

**Many movies won't have a franchise.** The LLM should output null/empty when
a movie is standalone. Most movies in the pipeline are not franchise films.

---

## Output Fields

Three fields per movie, stored in `franchise_membership`:

| Field | Type | Description |
|-------|------|-------------|
| `franchise_name_normalized` | TEXT NOT NULL | `normalize_string()` applied to the canonical IP name. The **only** stored form — no separate display-name column. |
| `franchise_role` | TEXT NOT NULL | `FranchiseRole` enum value (see below). Stored as integer ordinal. |
| `culturally_recognized_group` | TEXT, nullable | Stored normalized. Only when a culturally established grouping term exists **globally** (any market, not just US). Never hallucinated. |

### FranchiseRole enum

| Value | Meaning | Example |
|-------|---------|---------|
| `STARTER` | First entry / franchise originator | The Super Mario Bros. Movie (2023) for the Mario film franchise |
| `MAINLINE` | Numbered sequel or direct continuation | Sonic the Hedgehog 2, Harry Potter and the Chamber of Secrets |
| `SPINOFF` | Set in the same universe but following different characters/stories | Fantastic Beasts (Harry Potter spinoff), Puss in Boots (Shrek spinoff) |
| `PREBOOT` | Prequel that also serves as a continuity reboot | Batman Begins, Casino Royale (2006) |
| `REMAKE` | New version of a previous franchise entry | The Super Mario Bros. Movie (2023) relative to the 1993 film |

STARTER specifically marks franchise originators so the system can answer
"what started the X franchise."

### No display-name column

Only `franchise_name_normalized` is stored. Since it's just
`normalize_string()` applied to the canonical name, storing both the raw and
normalized forms is redundant. Display-form names can be derived at the UI
layer if ever needed. Same applies to `culturally_recognized_group` — stored
normalized.

### culturally_recognized_group — globally scoped

This field captures established grouping terminology for sub-sections of large
franchises. Examples: "original trilogy" (Star Wars), "MCU Phase 1" (Marvel),
"Daniel Craig era" (James Bond).

**Scope is global, not US-centric.** If a Bollywood trilogy, Korean franchise,
or Japanese anime series has a culturally established grouping name in its home
market, use it. The bar is "does an established name exist anywhere", not "does
an American know it."

**Conflict resolution:** In the rare case where multiple names exist across
markets for the same grouping, prefer the American-market term.

**Most movies get null.** Only a small number of large franchises have
established sub-groupings. The LLM is explicitly instructed to never
hallucinate a grouping name — null is correct when no established term exists.

---

## LLM Inputs

Seven fields, chosen for high signal with minimal token cost and no
generated-metadata dependencies:

| Input | Source | Typical tokens | What it reveals |
|-------|--------|---------------|-----------------|
| `title` | tmdb_data.title | ~5-10 | Primary identification signal. Most franchise movies are identifiable from title alone. |
| `release_year` | tmdb_data.release_date (extracted) | ~4 | Disambiguates when multiple movies share a title. |
| `overview` | imdb_data.overview | ~50-150 | **Identification aid only.** Helps the LLM correctly identify which movie this is when the title is ambiguous. Sometimes explicitly mentions "based on the video game", "sequel to...", etc. |
| `collection_name` | tmdb_data.collection_name | ~5-15 | Direct TMDB franchise signal. Present for ~25% of movies. When present, almost always correct. |
| `production_companies` | imdb_data.production_companies | ~10-30 | Brand-level franchise identification. "Marvel Studios" → MCU. "Illumination" → context for IP-based animated films. |
| `overall_keywords` | imdb_data.overall_keywords | ~10-40 | Often contain franchise names directly (e.g., "marvel-cinematic-universe", "james-bond", "based-on-video-game"). |
| `characters` (first 5) | imdb_data.characters | ~10-25 | Critical for IP-based franchises. "Mario", "Luigi", "Princess Peach" → Mario franchise. "Optimus Prime" → Transformers. "Barbie" → Barbie. Capped at first 5 to avoid bloating with extras ("Bully #2", "Waitress", etc.) — billing order means the recognizable IP characters appear first. |

**Total input: ~100-300 tokens per movie.** Very compact compared to other
generation types.

### Why each input is included

- **title + release_year:** Core identifiers. The LLM's parametric knowledge
  is the primary source of franchise information — these are what it needs to
  access that knowledge. Title alone is ambiguous (multiple movies share
  titles across years).

- **overview:** Net positive as an identification aid but needs prompt
  guardrails. The risk is the LLM inferring franchise from *plot similarity*
  rather than actual IP ownership. The prompt must frame it as "use this to
  identify the movie, not to infer franchise from plot content."

- **collection_name:** When present, basically provides the answer directly.
  Acts as strong confirmation signal.

- **production_companies:** Essential for studio-level and brand-level
  disambiguation. Sony's Spider-Man vs. MCU Spider-Man.

- **overall_keywords:** High signal, low noise. IMDB keywords frequently
  contain franchise names as structured tags.

- **characters (first 5):** The single most important input for the broadened
  franchise definition. Without characters, the LLM would miss many IP-based
  franchises where the title doesn't contain the IP name. Capped at 5 because
  IMDB lists characters in billing order — the recognizable IP characters
  appear first, while deep credits are extras like "Bully #2" that add noise
  without franchise signal.

### Inputs explicitly excluded

| Input | Why excluded |
|-------|-------------|
| `plot_synopses` / `plot_summaries` | Too much text for this task. Franchise membership is about IP ownership, not plot. High token cost, risk of plot-similarity hallucination. |
| `directors` / `writers` / `actors` | Not franchise-relevant. A director working on multiple franchise films doesn't make those films part of the same franchise. |
| `featured_reviews` | Reviews rarely discuss franchise positioning explicitly. |
| `genres` | Zero franchise signal. |
| `source_material_types` (generated) | Tempting (video_game_adaptation hints at a gaming franchise), but the LLM already has overview + keywords + characters to figure this out. Adding a generated enum creates a cross-generation dependency for marginal signal. |
| `franchise_lineage` (source_of_inspiration) | Free-text, designed for vector embedding, creates dependency chain. Better to generate franchise cleanly from raw inputs. |
| `awards`, `budget`, `revenue`, `ratings` | Completely orthogonal to franchise identification. |

### Data loading

The current `MovieInputData` (in `schemas/movie_input.py`) does not include
`characters`, `production_companies`, or `collection_name`. Franchise
generation will need either:
- A dedicated data loading function that queries these fields from
  `imdb_data` and `tmdb_data`, or
- An extension to `MovieInputData` (less preferable — adds fields only one
  generator uses)

All three fields exist in the tracker DB:
- `imdb_data.characters` — JSON array of character name strings (billing
  order; take first 5 only)
- `imdb_data.production_companies` — JSON array of company name strings
- `tmdb_data.collection_name` — TEXT, nullable (requires Task #5: TMDB
  collection name capture to be completed first)

---

## Eligibility

**No eligibility gate. All movies are eligible.**

This is a deliberate departure from every other generation type in the
pipeline, justified by the fundamentally different nature of this task:

### Why no gate

1. **Identification, not analysis.** Other generators need rich content to
   *analyze* — plot text, reviews, observations. Franchise generation needs
   enough to *identify* the movie, then accesses the LLM's parametric
   knowledge. Title + year alone is sufficient for most franchise movies.

2. **Sparse input causes conservative nulls, not hallucination.** When the
   LLM can't identify a movie, it outputs null (no franchise). This is the
   correct behavior for an unidentifiable movie. Compare to plot_analysis,
   where sparse input causes the LLM to fabricate themes from nothing.

3. **False negatives are worse than false positives.** Skipping a legitimate
   franchise movie means permanently missing that retrieval signal. Running
   on a non-franchise movie produces a null output — no cost beyond the API
   call.

4. **The quality funnel has already filtered.** By Stage 6, every movie has
   passed IMDB quality scoring (8 weighted signals, hard gates on title-type
   and missing-text). They all have title, year, and substantial metadata.
   The scenario of "a movie with literally nothing but a title" doesn't exist
   in this pipeline.

5. **Cost is negligible.** ~100-300 input tokens, ~50 output tokens for most
   movies (null or simple franchise record). Running all ~100K movies is
   trivially cheap compared to other generators.

6. **No single non-title input is gate-worthy.** `collection_name` is absent
   for ~75% of movies. Characters are absent for some. Keywords are absent
   for some. None of these absences should prevent generation — they just
   mean the LLM relies more on title + parametric knowledge.

---

## Prompt Strategy

### Canonical naming convention

Both the franchise generation LLM and the search extraction LLM are
instructed to output the most common, fully expanded form of the franchise/IP
name — no abbreviations. Same convention as the lexical entity extractor for
person names.

- "MCU" → "Marvel Cinematic Universe"
- "HP" → "Harry Potter"
- "F&F" → "Fast and Furious"
- "Mario" → "Super Mario" (or whatever the canonical expanded IP name is)

This ensures both sides converge on the same canonical string without needing
alias tables. After extraction, `normalize_string()` is applied and trigram
similarity against `lex.lexical_dictionary` resolves to a `term_id`.

### Overview framing

The overview must be framed in the prompt as an **identification aid**:

> "Use the overview to help identify which movie this is. Do NOT infer
> franchise membership from plot similarity — a movie about a heist in space
> is not part of the Ocean's franchise just because the plot sounds similar."

### Franchise definition in prompt

The prompt must explicitly define franchise as any recognizable IP/brand from
any medium, with examples spanning film series, video games, toys, books,
comics, and TV. Without this, the LLM will default to a narrow "film sequel
series" definition and miss IP-based franchises.

### culturally_recognized_group instructions

The prompt must instruct:
- This field is globally scoped — use established grouping names from any
  market, not just American/English terminology
- Leave null when no established term exists (the default for most movies)
- If multiple names exist across markets for the same grouping, prefer the
  name most recognized in the US market
- Never invent a grouping name — only use terms that are already culturally
  established

### Null output handling

The prompt should explicitly state that most movies are standalone and should
produce null/empty output. This prevents the LLM from over-assigning franchise
membership to movies with tenuous connections.

---

## Pipeline Placement

### Wave independence

Franchise generation has **no dependency on Wave 1 or Wave 2 outputs.** All
inputs come from raw TMDB/IMDB data in the tracker DB. This means it can
run:
- In parallel with Wave 1 (plot_events + reception)
- As a standalone batch before, during, or after the existing wave structure
- As its own independent generation pass

### Batch pipeline integration

Uses the existing batch generation infrastructure:
- `generator_registry.py` — register franchise as a new generation type
- `request_builder.py` — build batch requests with franchise inputs
- `openai_batch_manager.py` — submit and poll batch jobs
- `result_processor.py` — parse and store results

### Output storage

- **Tracker DB:** `generated_metadata.franchise` column (JSON) — raw LLM
  output for crash-safe resumption
- **Postgres:** `franchise_membership` table (structured, for search)
- **Lexical index:** `franchise_name_normalized` → `lex.lexical_dictionary`
  → `lex.inv_franchise_postings`

### TMDB collection capture dependency

`collection_name` requires Task #5 (TMDB collection name capture) to be
completed first. If franchise generation runs before Task #5, it will operate
without that input — still functional but missing a strong confirmation
signal for ~25% of movies.

---

## Output Schema Design

Simple structured output — much simpler than most generation types since
there are only 3 fields and no complex nested structures.

```python
class FranchiseOutput(BaseModel):
    franchise_name: str | None = Field(
        default=None,
        description="The canonical, fully expanded name of the franchise/IP "
                    "this movie belongs to. None if standalone."
    )
    franchise_role: FranchiseRole | None = Field(
        default=None,
        description="The movie's role within the franchise."
    )
    culturally_recognized_group: str | None = Field(
        default=None,
        description="Established grouping term (any market, globally). "
                    "None for most movies."
    )
```

Note: `franchise_name` is the raw canonical name in the LLM output. It gets
normalized via `normalize_string()` before storage as
`franchise_name_normalized`. The raw form is not stored separately.

---

## Downstream Usage in Search

### Phase 0 (Query Understanding)

The search extraction LLM decomposes franchise queries into up to three
components:
- `franchise_name` — canonical expanded form, same naming convention
- `franchise_role` — same `FranchiseRole` enum definition
- `culturally_recognized_group` — free text qualifier

### Phase 1 (Candidate Retrieval)

1. `franchise_name` → `normalize_string()` → trigram similarity against
   `lex.lexical_dictionary` → `term_id`
2. Lexical lookup: `lex.inv_franchise_postings (term_id → movie_id)` →
   candidate set of 3-30 movies
3. Optional `franchise_role` filter: `WHERE franchise_role = $enum_ordinal`
4. Optional `culturally_recognized_group` match: trigram similarity on the
   normalized group column. No index needed — candidate set is 3-30 movies.

### Replaces

Current franchise search in `lexical_search.py` that combines title tokens +
character matching. The current approach matches too broadly — "Marvel"
matches via character names rather than franchise membership, creating the
"Franchise Logic Is Too Broad" problem documented in
`current_search_flaws.md` (Problem #8).

---

## Storage Schema

### Postgres table

```sql
CREATE TABLE IF NOT EXISTS public.franchise_membership (
    movie_id                        BIGINT NOT NULL REFERENCES movie_card,
    franchise_name_normalized       TEXT NOT NULL,
    culturally_recognized_group     TEXT,       -- normalized; globally scoped
    franchise_role                  TEXT NOT NULL,
    PRIMARY KEY (movie_id, franchise_name_normalized)
);
```

### Lexical posting table

```sql
CREATE TABLE IF NOT EXISTS lex.inv_franchise_postings (
    term_id   BIGINT NOT NULL,
    movie_id  BIGINT NOT NULL,
    PRIMARY KEY (term_id, movie_id)
);
```

`franchise_name_normalized` is inserted into `lex.lexical_dictionary` and the
posting table maps `term_id → movie_id` for text-based franchise lookup.

---

## Dependency on Other V2 Work

- **Requires:** Task #5 (TMDB collection name capture) for optimal input
  quality, but can run without it
- **Does NOT depend on:** Wave 1 outputs, Wave 2 outputs, keyword audit,
  source material enum, country enum, awards scraping, concept tags, or any
  other generation type
- **Blocked by nothing** — can start immediately with existing tracker data
  (minus collection_name)

---

## Open Questions

None currently. All design decisions have been made:
- Output fields: decided (3 fields, normalized-only storage)
- LLM inputs: decided (7 fields, no generated-metadata dependencies)
- Eligibility: decided (no gate)
- Franchise definition: decided (any IP/brand from any medium)
- culturally_recognized_group scope: decided (global with US precedence)
- Canonical naming: decided (fully expanded, no abbreviations)
- Pipeline placement: decided (wave-independent, uses batch infrastructure)
