# Franchise Metadata Generation

LLM-generated franchise data for each movie. Classifies a film along two
orthogonal axes (identity + narrative position) plus a top-level
franchise-launch flag, so the search system can answer the full range of
franchise queries — "what X movies are there?", "movies inside the X
subgroup", "movies that kicked off a franchise", "sequels vs reboots vs
spinoffs of X", and "crossovers between X and Y".

Replaces the current title-token + character-matching franchise heuristic in
lexical search with a dedicated `movie_franchise_metadata` table and
`lex.inv_franchise_postings` posting table.

## Version history

- **v1 / v2** (historical): Three flat fields — `franchise_name`,
  `franchise_role` (closed enum: STARTER/MAINLINE/SPINOFF/PREBOOT/REMAKE),
  `culturally_recognized_group`. Deprecated — the closed `franchise_role`
  enum conflated identity with narrative position and couldn't represent
  pair-remakes, first-entry-in-lineage, or reboot/remake overlap.
- **v3** (historical): Same shape with tightened `franchise_role` semantics.
  Still suffered from axis conflation.
- **v4**: Two-axis rewrite. IDENTITY axis (lineage, shared_universe,
  recognized_subgroups, launches_subgroup) and NARRATIVE POSITION axis
  (lineage_position enum, special_attributes enum array). Fixed axis
  conflation.
- **v5** (current): Adds `launched_franchise` top-level flag, global
  normalization rule for every named entity, shape-B `shared_universe` for
  spinoff-parent relationships, field rename `launches_subgroup` →
  `launched_subgroup`. See
  [franchise_test_iterations.md](franchise_test_iterations.md) for the full
  v3→v4→v5 rationale and worked examples.

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

## Output Fields (v5)

Seven fields per movie, organized by axis. See
[schemas/metadata.py::FranchiseOutput](../schemas/metadata.py) for the
Pydantic contract and
[movie_ingestion/metadata_generation/prompts/franchise.py](../movie_ingestion/metadata_generation/prompts/franchise.py)
for the generation prompt.

### IDENTITY axis — what brands/groups the film belongs to

| Field | Type | Description |
|-------|------|-------------|
| `lineage` | TEXT, nullable | **Narrowest** recognizable line of films this entry descends from — `batman`, `spider-man`, `harry potter`, `godzilla`, `iron man`. Null when standalone or no multi-entry line exists. |
| `shared_universe` | TEXT, nullable | Broader entity above the lineage, under TWO shapes: (A) formal studio-recognized shared cinematic universe hosting multiple lineages (`marvel cinematic universe`, `dc extended universe`, `wizarding world`, `monsterverse`, `conjuring universe`); (B) parent franchise of a spinoff sub-lineage (Puss in Boots → `shrek`, Minions → `despicable me`, Logan → `x-men`). Null when the lineage is itself the top-level brand. |
| `recognized_subgroups` | TEXT[] | Named sub-phases this film belongs to (`phase one`, `infinity saga`, `the dark knight trilogy`, `daniel craig era`, `kelvin timeline`). EMPTY-THEN-ADD framing — empty list is the most common outcome. |
| `launched_subgroup` | BOOL | True iff this film is the earliest-released entry in at least one of its recognized_subgroups. |

### NARRATIVE POSITION axis — how this film relates to prior films

| Field | Type | Description |
|-------|------|-------------|
| `lineage_position` | ENUM, nullable | Mutually exclusive: `sequel` / `prequel` / `remake` / `reboot` / `null`. Null for first entry in lineage OR standalone. May populate even when `lineage` is null (pair-remakes like Scarface 1983). |
| `is_spinoff` | BOOL | Orthogonal flag: derivative work that expands a minor character, story element, or subplot from an existing lineage into a new film that leaves the source behind. |
| `is_crossover` | BOOL | Orthogonal flag: two or more distinct top-level lineages combined into a single film. |

### FRANCHISE LAUNCH flag — cinematic origin of a new franchise

| Field | Type | Description |
|-------|------|-------------|
| `launched_franchise` | BOOL | True iff this film is the cinematic origin of a franchise audiences today recognize as a multi-film franchise, per a four-part test (see below). Distinct from `launched_subgroup`. |

### Axis independence

The three blocks are independent. A film can carry a lineage_position even
when lineage is null (pair-remakes). A film can be both a sequel AND a
spinoff (Puss in Boots: The Last Wish continues the Puss in Boots
sub-lineage AND is a spinoff of Shrek). `launched_franchise` and
`launched_subgroup` can fire independently — Iron Man (2008) fires only
`launched_subgroup=true` (Marvel franchise already existed), Shrek (2001)
fires only `launched_franchise=true` (no named subgroup inside shrek),
Star Wars: A New Hope (1977) fires both (launched the franchise AND the
original trilogy subgroup).

### launched_franchise — four-part test

All four tests must pass for `launched_franchise` to be true:

1. **First cinematic entry** — `lineage_position` must be null.
2. **Not a spinoff** — `is_spinoff` must be false.
3. **Source-material recognition test** — if adapted, the audience must
   recognize THE FILM (or film franchise) MORE than any prior book / game
   / toy / cartoon / show. Jurassic Park passes (novel obscure); Harry
   Potter 1 fails (books dominate); Barbie fails (toy line dominates).
4. **Relevant follow-ups test** — the film must have spawned sequels,
   prequels, reboots, or spinoffs that audiences recognize as a
   continuing franchise. Jaws fails (forgotten sequels); Forrest Gump
   fails (no follow-ups).

Silently corrected by `FranchiseOutput.validate_and_fix()` when any
precondition fails (lineage null, lineage_position populated, or
`is_spinoff=true`).

### LineagePosition enum

| Value | Meaning | Example |
|-------|---------|---------|
| `sequel` | Continues existing continuity forward. Includes legacy sequels with returning protagonists. | The Dark Knight, Empire Strikes Back, Creed, Top Gun: Maverick |
| `prequel` | Set chronologically before an earlier-released film in the same lineage, with shared continuity. Reboots set early are NOT prequels. | The Hobbit, Rogue One, Monsters University, Solo |
| `remake` | Retells the core story of a specific prior film with fresh production. Same story spine, different cast/period. Legal even when lineage is null. | Scarface (1983), The Lion King (2019), True Grit (2010) |
| `reboot` | Restarts an existing lineage's continuity with a NEW story. Requires ≥1 prior theatrical entry in the lineage. | Batman Begins, Casino Royale (2006), The Amazing Spider-Man, Star Trek (2009) |
| `null` | First entry in lineage, or standalone. | Iron Man (2008), Fellowship of the Ring, Harry Potter 1 |

**Note:** The `remake` value is retained in the enum for classification
fidelity but is NOT consumed at search time — `source_of_inspiration` covers
film-to-film retellings more uniformly (including cross-medium adaptation).
Documented via code comment above the enum member in
[schemas/enums.py](../schemas/enums.py).

### Narrative-position booleans

| Field | Meaning |
|-------|---------|
| `is_spinoff` | Derivative work that expands a MINOR character, story element, or subplot from an existing lineage into the focus of a new film that LEAVES THE SOURCE FILM'S MAIN CHARACTERS AND PLOT BEHIND. |
| `is_crossover` | Two or more distinct top-level lineages combined into a single film with characters from both. |

### Global normalization rule (v5)

Every named entity emitted — `lineage`, `shared_universe`, and each
`recognized_subgroups` label — is normalized identically:

- Lowercase everything.
- Spell digits as words (`phase 3` → `phase three`).
- Expand `&` to `and` (`Fast & Furious` → `fast and furious`).
- Expand abbreviations and first+last names **only when the expansion is
  itself in common use** (`MCU` → `marvel cinematic universe` ✓;
  `monsterverse` stays as `monsterverse`).
- Drop first names on director-era labels where surname alone is the
  common form (`sam raimi trilogy` → `raimi trilogy`).
- Casing of proper nouns handled downstream in display code — the
  generation and storage layer is lowercase.

This is enforced by a top-level GLOBAL OUTPUT RULES block in the prompt
and silently restated inside FIELD 3+4 for emphasis.

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

## Output Schema Design (v5)

Reasoning-before-answer field ordering — each decision field is paired
with a scoped reasoning field that must be produced first (chain-of-
thought via schema order). Descriptions are intentionally compact; the
system prompt carries the definitional weight.

```python
class FranchiseOutput(BaseModel):
    # Identity block
    lineage_reasoning: str
    lineage: str | None                           # narrowest line, normalized
    shared_universe: str | None                   # shape A or shape B, normalized

    # Subgroup block
    subgroups_reasoning: str
    recognized_subgroups: list[str]               # normalized labels
    launched_subgroup: bool                       # coupled to recognized_subgroups

    # Narrative position block
    position_reasoning: str
    lineage_position: LineagePosition | None      # mutually exclusive enum
    special_attributes: list[SpecialAttribute]    # multi-valued enum

    # Franchise launch flag
    launch_reasoning: str
    launched_franchise: bool                      # four-part test, validator-fixed
```

`validate_and_fix()` enforces internal consistency after parsing: partial
null-propagation (if lineage null, clear shared_universe /
recognized_subgroups / launched_subgroup; preserve lineage_position and
special_attributes for pair-remakes and standalone spinoff-flavored
films), `launched_subgroup ⇄ recognized_subgroups` coupling,
`launched_franchise` coherence (force false when lineage null,
lineage_position populated, or spinoff in special_attributes), and
`special_attributes` dedup.

---

## Downstream Usage in Search (v5)

### Phase 0 (Query Understanding)

The search extraction LLM decomposes franchise queries into any of:

- `lineage` — canonical expanded form, same global normalization rule as
  generation. Always lowercase. Example: "dark knight movies" → `batman`
  with subgroup filter.
- `shared_universe` — same normalization. Searchable via shape A (formal
  cosmos) OR shape B (parent franchise of a spinoff). "spinoffs of shrek"
  hits `shared_universe = shrek`.
- `recognized_subgroups` — free-text qualifier, trigram-matched against
  stored normalized labels. "mcu phase 3 movies" → lineage=any marvel
  lineage, subgroup=`phase three`.
- `lineage_position` — enum filter. "batman sequels" → lineage=batman,
  lineage_position=sequel. NOTE: `remake` is NOT consumed at search time
  — use `source_of_inspiration` for cross-film retelling queries.
- `is_spinoff` / `is_crossover` — boolean filters. "marvel spinoffs" →
  shared_universe=`marvel cinematic universe` and `is_spinoff=true`.
- `launched_subgroup` — boolean filter. "first entry in phase three"
  combines subgroup=`phase three` with launched_subgroup=true.
- `launched_franchise` — boolean filter. "movies that launched a
  franchise" → launched_franchise=true.

### Phase 1 (Candidate Retrieval)

1. `lineage` / `shared_universe` → `normalize_string()` → trigram similarity
   against `lex.lexical_dictionary` → `term_id` → shared franchise posting lookup
2. Boolean filters (`launched_subgroup`, `launched_franchise`,
   `is_spinoff`, `is_crossover`) and `lineage_position` applied as
   structured WHERE clauses on `movie_franchise_metadata`
4. Optional `recognized_subgroups` match: trigram similarity on the
   normalized labels. Candidate set is small enough that no index is
   needed.

The two-axis design means a single query like "movies that launched
a franchise" is a single structured filter (`launched_franchise=true`),
while "spinoffs of marvel" is two filters (shared_universe=`marvel
cinematic universe` AND `is_spinoff=true`), with no
axis-conflation ambiguity.

### Replaces

Current franchise search in `lexical_search.py` that combines title tokens
+ character matching. The current approach matches too broadly — "Marvel"
matches via character names rather than franchise membership, creating the
"Franchise Logic Is Too Broad" problem documented in
`current_search_flaws.md` (Problem #8). The v5 two-axis + launched_franchise
design also unlocks query types that v1-v3 could not express
structurally: "movies that launched a franchise", "spinoffs of [parent
franchise]", "first entry in [subgroup]", and "sequels vs reboots of X"
with unambiguous enum filtering.

---

## Storage Schema (implemented)

The current Postgres projection stores the finalized franchise object in a
single table, and indexes both `lineage` and `shared_universe` into the same
franchise posting table for tolerant lookup.

```sql
CREATE TABLE IF NOT EXISTS public.movie_franchise_metadata (
    movie_id               BIGINT PRIMARY KEY REFERENCES public.movie_card ON DELETE CASCADE,
    lineage                TEXT,              -- normalized; nullable
    shared_universe        TEXT,              -- normalized; nullable
    recognized_subgroups   TEXT[] NOT NULL DEFAULT '{}',
    launched_subgroup      BOOLEAN NOT NULL DEFAULT FALSE,
    lineage_position       SMALLINT,          -- LineagePosition ordinal; nullable
    is_spinoff             BOOLEAN NOT NULL DEFAULT FALSE,
    is_crossover           BOOLEAN NOT NULL DEFAULT FALSE,
    launched_franchise     BOOLEAN NOT NULL DEFAULT FALSE
);
```

`recognized_subgroups` stays inline as a text array. `launched_subgroup=true`
is interpreted as "this movie launched all listed subgroups" — acceptable
over-counting is preferred to under-counting.

### Lexical posting table

```sql
CREATE TABLE IF NOT EXISTS lex.inv_franchise_postings (
    term_id   BIGINT NOT NULL,
    movie_id  BIGINT NOT NULL,
    PRIMARY KEY (term_id, movie_id)
);
```

Both `lineage` and `shared_universe` are inserted into
`lex.lexical_dictionary` and share the same posting table. This is
deliberately tolerant: lookup should succeed whether the LLM put the
user-visible brand in `lineage` or `shared_universe`, and whether a
parent-universe label like `shrek` is acting as the lineage or the
umbrella for a spinoff branch.

Important nuance: `term_id` here is only the integer ID of the normalized
string in `lex.lexical_dictionary`, not a separate franchise-entity ID. That
storage shape is fine, but query-time franchise resolution must NOT rely on
exact string lookup alone. `lineage` and `shared_universe` are often known
enough to normalize consistently, but not reliably enough to assume exact
LLM string reproduction for every query (`mcu` vs `marvel cinematic universe`,
`wizarding world` vs `harry potter universe`, etc.). The intended retrieval
path is fuzzy resolution of the user phrase against franchise strings first,
then posting lookup by the resolved `term_id`.

`recognized_subgroups` are intentionally NOT inserted into
`lex.inv_franchise_postings`. Subgroup labels have much higher string
variability than lineage/shared-universe names, so they stay inline on
`movie_franchise_metadata` and are matched only after the franchise lookup
has already narrowed the candidate set.

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

## Open Questions (v5 state)

### Resolved in v5

- **Axis design**: decided (identity + narrative position + launch flag,
  three independent blocks).
- **Output fields**: decided (seven per-movie fields, documented in the
  Output Fields section above).
- **LLM inputs**: decided in v4 (title_with_year, release_year, overview,
  collection_name, production_companies, directors, overall_keywords,
  characters, top_billed_cast — labeled with their role in reasoning).
- **Eligibility**: decided (no gate, identification-not-analysis task).
- **Franchise definition**: decided (any IP/brand from any medium, though
  the v5 narrowest-lineage flip means the STORED `lineage` is the specific
  line of films — `batman`, not `dc comics`).
- **Canonical naming**: decided (global normalization rule — lowercase,
  digits as words, `&` → `and`, expand only when the expansion is in
  common use).
- **Shared universe shape B**: decided (spinoff-parent relationships
  populate shared_universe with the parent franchise).
- **launched_franchise four-part test**: decided (first cinematic entry,
  not a spinoff, source-material recognition, relevant follow-ups).
- **REMAKE handling**: decided (retained in enum, NOT read at search time;
  `source_of_inspiration` covers film-to-film retellings).
- **Pipeline placement**: decided (wave-independent, uses batch
  infrastructure).

### Still open

- **Storage schema ADR**: the v5 draft above needs to be reviewed and
  promoted to an ADR before implementation. Key decisions: subgroups in a
  side table vs array column, special_attributes as array vs side table,
  posting-list strategy for launched_franchise / launched_subgroup.
- **Spinoff redefinition**: deferred by user to a future conversation —
  the v4/v5 three-constraint test (MINOR IN SOURCE, GOES SOMEWHERE NEW,
  LEAVES THE SOURCE BEHIND) is intact but under review.
- **X-Men (2000) subgroup**: unlike Spider-Man (2002) which now has
  `raimi trilogy`, X-Men (2000) still has empty `recognized_subgroups`.
  Does `original x-men trilogy` or `fox x-men films` meet the real-world
  usage bar? To be decided once we see how the model handles Spider-Man
  in the v5 rerun.
- **Regional launched_franchise ambiguity**: Ip Man and K.G.F Chapter 1
  are ruled TRUE as judgment calls. If the v5 model disagrees, we may
  need a clarifying bullet in the test-3 recognition-test section about
  regional cinema.

---

## Next Test Run — what we want to learn

After the v5 rerun against the same four candidate models
(gpt5-mini-medium, gpt5-mini-low, gpt5-mini-minimal, gpt54-mini-low) on
the 79-movie SOT, we are specifically evaluating:

### Primary questions

1. **Does the GLOBAL OUTPUT RULES normalization block actually hold?**
   Baseline measurement: v4 had zero normalization on lineage /
   shared_universe. In v5 every emitted entity should be lowercase, with
   `&` → `and`, digits spelled out, and expansions only when in common
   use. Failure modes to watch: MCU / DCEU leaking into lineage as
   compact forms, Title Case slippage, digit retention in titles like
   `fast 2 furious`, inconsistent expansion where the same model emits
   `mcu` in one field and `marvel cinematic universe` in another.

2. **Does shape-B `shared_universe` fire correctly?** We expect
   `shared_universe = shrek` on Puss in Boots, `shared_universe =
   despicable me` on Minions, `shared_universe = x-men` on Logan. v4
   would reject all three. Watch for: (a) the model keeping the old v4
   "formal cosmos only" interpretation, (b) the model over-applying
   shape B to non-spinoff cases, (c) Hobbs & Shaw — does the model
   correctly keep `lineage = fast and furious` with no shared_universe
   (per SOT), or does it try to invent a `hobbs and shaw` lineage?

3. **Does the `launched_franchise` four-part test discriminate the
   Group C cases correctly?** The critical distinction: Iron Man 2008,
   Batman Begins, Casino Royale, Man of Steel should all fire
   `launched_subgroup=true` but `launched_franchise=false` (source
   material / pre-existing franchise dominates). Shrek, Matrix, Saw,
   Jurassic Park, Mad Max, Ip Man, K.G.F should fire
   `launched_franchise=true`. Failure modes to watch:
   - Model flipping both flags true on Group C (conflates subgroup launch
     with franchise launch).
   - Model flipping both flags false on starters (too conservative).
   - Model firing `launched_franchise=true` on Jaws or Forrest Gump
     (ignoring the relevant-follow-ups test).
   - Model firing `launched_franchise=true` on Barbie or Super Mario Bros.
     (ignoring the source-material recognition test).
   - Model firing `launched_franchise=true` on spinoffs (coherence
     violation — should be caught by `validate_and_fix()` but we want to
     see how often the LLM gets it right before the guardrail kicks in).

4. **Does the anti-restatement carveout work for director/actor eras?**
   Sherlock Holmes (2009) should fire `launched_subgroup=true` with
   `ritchie sherlock holmes films` as the subgroup. Spider-Man (2002)
   should fire `launched_subgroup=true` with `raimi trilogy`. The v4
   anti-restatement rule would have stripped these as "bare
   restatements" of the lineage. Watch for labels that were previously
   blocked now slipping back through incorrectly.

5. **How does reasoning effort (medium vs low vs minimal) trade against
   accuracy on the new launched_franchise field?** v4 analysis showed
   medium ≈ low on the other fields, which is why production runs
   `reasoning_effort: low`. If launched_franchise specifically needs
   medium, that's a reason to bump the production setting.

### Secondary questions

- Does the Rogue One subgroup `star wars anthology films` actually get
  recognized by the model, or does it fall below the
  culturally-used-label bar? If the model rejects it, we reconsider the
  subgroup and fall back to `launched_subgroup=false` for the Star Wars
  anthologies.
- On the `MCU` / `marvel cinematic universe` expansion specifically: the
  prompt's normalization examples explicitly show this expansion. Do
  any candidate models still emit `mcu`?
- Does `validate_and_fix()`'s new `launched_franchise` coherence block
  ever fire on a run where the LLM's reasoning itself was right? i.e.
  are we silently correcting real errors, or is the guardrail a safety
  net the model rarely needs?
- Does the `lineage_position` field stay stable under the rename? No
  changes were made to that field — regressions here would indicate
  cross-field interference from the global normalization rule.

### Go / no-go criteria for v5 → production

- `launched_franchise` accuracy ≥ 85% across the 79-row SOT on at least
  one candidate model.
- Normalization rule holds on ≥ 95% of emitted entities (no compact
  forms leaking into lineage / shared_universe).
- Shape-B `shared_universe` passes on at least 3/4 spinoff-parent cases
  (Puss in Boots, Minions, Logan, plus any additional cases uncovered in
  the rerun).
- No regression on v4-level accuracy for `lineage`, `lineage_position`,
  or `special_attributes` — v5 should not have cost us anything on the
  fields that were already working.
- Group C discrimination: Iron Man / Batman Begins / Casino Royale /
  Man of Steel all get `launched_franchise=false` across ≥ 90% of runs.

If v5 passes, the next step is ADR authoring for the storage schema and
migration of the ingestion writer to produce the new seven-field output.
If any go/no-go criterion fails, we iterate on the prompt before
promoting v5 to batch generation.
