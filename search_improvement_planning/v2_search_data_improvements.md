# V2 Search Data Improvements: Studio Entity Resolution

## Context

The v1 search treats studio lookups as exact-match against
`lex.lexical_dictionary` after string normalization. This fails
systematically for the dominant real-world pattern: users name a
brand ("Disney", "MGM", "A24") and the DB stores several
variants per brand, including subsidiaries whose names share no
tokens with the parent ("Pixar", "Marvel Studios", "HBO",
"Lucasfilm").

The DB currently holds ~181,871 unique production company
strings, derived from `imdb_data.production_companies`. Empirical
analysis surfaced eleven distinct failure modes documented in
the "Studio Matching — Full Problem Dimensions" section below.
This design addresses them through three coordinated mechanisms:

1. **A closed brand registry** (~50 major brands) with
   time-bounded member companies, stamped onto each movie at
   ingest based on the movie's release year. Handles umbrella
   queries ("Disney"), time-bounded ownership (Lucasfilm 1977 vs
   2015), and historical renames (Fox → 20th Century Studios).
2. **Query-time LLM canonicalization** into a structured spec
   with either a `brand_id` (closed enum) or up to three
   `freeform_names` (surface forms likely to appear in IMDB).
   The LLM never handles ownership structure or cross-script
   translation — it emits surface forms the data already
   contains.
3. **A token-inverted index** over normalized
   `production_company` strings, used for freeform matching of
   specific sub-labels and long-tail studios.

## Design Principles

1. **Decouple LLM task from data mapping.** The LLM emits either
   a closed-enum `brand_id` or a short list of natural-language
   surface forms. Code resolves to production_company IDs and
   then to movies. The LLM is never asked to know the exact DB
   string or ownership history.
2. **Specificity preserved by default.** Multi-token freeform
   names narrow via posting-list intersection. "Walt Disney
   Animation" returns only animation subsidiaries, not all of
   Disney.
3. **Time-bounded brand membership is data, not a query-time
   decision.** Each movie is stamped at ingest with the brand
   IDs that owned its production companies *at the movie's
   release year*. "Disney" queries return Star Wars 2015 but
   not Star Wars 1977.
4. **Automated bulk, minimal curation.** The brand registry is
   ~50 rows (seeded from Wikidata). Everything else is
   auto-built. No 300-cluster editorial review, no 100-entry
   curated alias table.
5. **Long tail is first-class.** Production companies with no
   brand membership are matched via token intersection. Queries
   for `Villealfa Filmproductions`, `Carolco`, `Cannon Films`
   work via singleton tokens without any curation.

## Data Model

```
brand
  id PK          -- closed enum (BrandEnum): DISNEY, WARNER, A24,
                    MGM, STUDIO_GHIBLI, SONY, UNIVERSAL, ...
  canonical_name

brand_member_company
  brand_id FK
  production_company_id FK
  start_year SMALLINT NULLABLE   -- null = from inception
  end_year   SMALLINT NULLABLE   -- null = to present
  PRIMARY KEY (brand_id, production_company_id, start_year)

production_company
  id PK
  canonical_string
  normalized_string

studio_token
  token
  production_company_id FK
  PRIMARY KEY (token, production_company_id)

-- Per-movie stamp (on the movie row):
movie.brand_ids INT[]   -- empty when no brand membership applies
```

**brand** — closed registry of ~50 rows. Seeded from Wikidata's
`film production company` class plus manual additions. Curation
is deferred to a separate work stream (see Open Decisions).

**brand_member_company** — time-bounded membership edges. One
row per (brand, company, ownership era). Null `start_year` /
`end_year` means "always". Multiple rows per company are legal:
Miramax has a Disney-era row (1993–2010) and a non-Disney-era
row (2010–). A company can also be a member of multiple brands
simultaneously — e.g., Pixar is its own brand AND a member of
the Disney brand since 2006, so narrow "Pixar" queries and
umbrella "Disney" queries both work.

**production_company** — source of truth for every distinct
canonical IMDB string (~181K rows). `normalized_string` is
produced by `normalize_string` plus the new ordinal
number-to-word rule (see Normalization Rules). No `brand_id`
column — membership lives on `brand_member_company` with time
bounds.

**studio_token** — inverted index of `(token,
production_company_id)` pairs after stoplist and DF-ceiling
filtering. Tokenization splits on whitespace AND hyphens. GIN
index on `token` for fast posting-list lookup. Secondary index
on `production_company_id` for debugging.

**movie.brand_ids** — the ingest-time stamp. Set once per movie
at ingest by resolving each `production_companies` string to a
`production_company_id`, looking up `brand_member_company` rows
where the movie's release year falls within `[start_year,
end_year]`, and unioning the resulting brand IDs. GIN-indexed.

## Ingestion Pipeline

All stages are deterministic, crash-safe, and re-runnable.

### Stage A — Populate `production_company`

Sweep `imdb_data.production_companies` for every distinct
canonical string across all movies. Insert one row per unique
string with `normalized_string = normalize_string(canonical_string)`.
Upsert semantics so re-running adds new studios without
disturbing existing rows.

### Stage B — Compute Token Document Frequencies

For every row in `production_company`:

1. Tokenize `normalized_string` on whitespace AND hyphens.
2. Drop tokens in the domain boilerplate stoplist:
   `{pictures, picture, studios, studio, films, film,
     entertainment, productions, production, company, co,
     corporation, corp, inc, ltd, llc, group, media,
     cinéma, cinema, filmproductions, filmproduction,
     tv, television, broadcasting, network, networks,
     international, global, home, video}`
   (refine empirically from Stage C bucket results).
3. For each surviving token, increment a DF counter (one per
   distinct production_company — do not double-count repeats
   within a single string).
4. Write `(token, df)` to a working `token_frequency` table.

DF is measured per canonical string, not per movie.

### Stage C — Select the DF Ceiling

Run the bucket analysis described in **DF Ceiling Determination**
(below). Pick a value that cleanly separates discriminative
brand tokens (warner, disney, ghibli, ufa) from residual
boilerplate. Starting guess: **500**.

### Stage D — Build `studio_token`

For every row in `production_company`, tokenize again (same
rules as Stage B), filter by `df <= ceiling`, and insert one row
into `studio_token` per `(token, production_company_id)` pair.
**DF floor is 1** — singletons included, which is what makes
the long tail findable.

Create a GIN index on `studio_token.token`.

### Stage E — Brand Membership Stamping

For each movie:

1. Resolve each raw `production_companies` string to a
   `production_company_id` via `normalized_string` lookup.
2. For each production_company_id, find `brand_member_company`
   rows where `movie.release_year BETWEEN start_year AND end_year`
   (nulls interpreted as unbounded).
3. Write the union of matched brand_ids to `movie.brand_ids`.

Runs at movie-ingest time (one-shot per movie). Re-computed only
when the brand registry changes, and only for affected movies.

## Query-Time Resolution

The stage-3 query-time LLM emits a `StudioQuerySpec`:

```python
class StudioQuerySpec(BaseModel):
    thinking: str                       # reasoning about user intent,
                                        # scope (umbrella vs. specific),
                                        # and whether a closed-registry
                                        # brand applies
    brand_id: Optional[BrandEnum]       # closed enum; set ONLY for
                                        # umbrella queries
    freeform_names: list[str]           # up to 3: condensed /
                                        # expanded / other-common-form,
                                        # each a surface form likely
                                        # to appear in IMDB
```

**Schema ordering is load-bearing.** The `thinking` field comes
first so the LLM reasons about scope and brand applicability
*before* committing to `brand_id` or `freeform_names`. This
mirrors the pattern used elsewhere in stage-3 translators where
a reasoning preface improves downstream field quality.

**Prompt contract:**

- Set `brand_id` only when the user is asking at the umbrella /
  parent level ("disney", "warner", "sony movies", "mgm
  musicals").
- Leave `brand_id` null and rely on `freeform_names` when the
  user names a specific sub-label ("walt disney animation",
  "pixar", "hbo documentary", "fox searchlight").
- Each `freeform_name` must be a surface form that plausibly
  appears in IMDB's `production_companies` convention. Do NOT
  emit semantic translations ("japan broadcasting corporation"
  for NHK) unless the translation is a form IMDB actually uses.
- Emit up to 3 names covering: (1) the condensed / acronym form,
  (2) the expanded form, (3) another well-known alternative.
  Emit fewer if only one or two well-known forms exist.
- **Streamer disambiguation.** For brands whose name doubles as a
  streaming platform (currently `NETFLIX`, `AMAZON_MGM`,
  `APPLE_STUDIOS`), set `brand_id` to that brand **only** when the
  user's intent is clearly about *production* ("Netflix
  originals", "movies made by Netflix", "Apple original films",
  "Prime Video originals"). If the user's intent is *streaming
  availability* ("movies on Netflix", "streaming on Apple TV+",
  "what's on Prime"), leave the studio channel's `brand_id` null
  and emit no freeform_names covering the streamer — the
  `watch_providers` metadata path handles that intent. The
  `thinking` field is where this distinction is worked out before
  `brand_id` is committed.

### Step 1 — Path Selection

- If `brand_id` is set → Brand Path.
- Else → Freeform Path.

### Step 2a — Brand Path

```sql
SELECT movie_id FROM movie WHERE brand_ids && ARRAY[:brand_id]
```

Returns movies time-accurately stamped with that brand at ingest.
Star Wars 1977 (Lucasfilm pre-Disney) does not match
`brand_id = DISNEY`. Force Awakens 2015 does.

### Step 2b — Freeform Path

For each name in `freeform_names`:

1. Apply `normalize_string` (including number-to-word ordinal
   conversion).
2. Tokenize on whitespace AND hyphens.
3. Drop tokens matching the boilerplate stoplist OR with
   `DF > ceiling`.
4. If all tokens are dropped, this name contributes nothing.
5. Otherwise, for each surviving token, fetch its posting list
   from `studio_token`.
6. **Intersect** posting lists across the tokens within this
   name (AND semantics). Result: production_company_ids whose
   normalized_string contains every discriminative token of
   this name.

Then across names:

7. **Union** the per-name production_company_id sets.
8. Resolve to canonical strings; feed to the existing lexical
   movie posting-list retrieval.

**Matching semantics summary:**
- Within a single `freeform_name` → **intersection** (all
  tokens must match a single company string)
- Across multiple `freeform_names` → **union** (matching any
  one name counts as a hit)

### Step 3 — No Further Fallbacks

If both paths return empty, the studio channel contributes
nothing. Vector and metadata channels carry the search. No
fuzzy, trigram, or embedding fallback.

## Normalization Rules

Applied symmetrically at ingest and query time:

1. Lowercase
2. Diacritic-fold (NFKD → strip combining marks)
3. Strip punctuation (`.`, `,`, `:`, `;`, `'`, `"`, `(`, `)`,
   `/`, `+`, etc.)
4. Collapse whitespace
5. **Ordinal number-to-word**: `20th` → `twentieth`, `21st` →
   `twenty-first`, `1st` → `first`, `2nd` → `second`, ... This
   makes `20th Century Fox` (19 tags) match `Twentieth Century
   Fox` (1,361 tags), the dominant form in IMDB.

**Tokenization splits on whitespace AND hyphens.** This differs
from any title-side tokenizer that does not hyphen-split — flag
to keep in mind when auditing normalization symmetry.
`Metro-Goldwyn-Mayer` tokenizes to `{metro, goldwyn, mayer}`,
which lets a user typing "metro goldwyn mayer" match
`Metro-Goldwyn-Mayer Cartoon Studios` (a variant that has no
embedded "mgm" token).

**Out of scope:** Roman numerals (no major studio uses them
in a way that affects retrieval), compound-word splitting
(IMDB is consistent one-word on `DreamWorks`, etc.),
abbreviation expansion beyond ordinals.

## DF Ceiling Determination

The DF ceiling is the single most important empirical parameter.
Too low and real brand tokens like "warner" or "disney" get
excluded, making common queries fail. Too high and weakly
discriminative residual tokens pollute posting-list intersections
with noise.

Before committing to a value, run this analysis at the end of
Stage B:

```sql
SELECT
  CASE
    WHEN df >= 10000 THEN '>=10000'
    WHEN df >= 2000  THEN '2000-9999'
    WHEN df >= 500   THEN '500-1999'
    WHEN df >= 200   THEN '200-499'
    WHEN df >= 50    THEN '50-199'
    WHEN df >= 10    THEN '10-49'
    WHEN df >= 2     THEN '2-9'
    ELSE '1'
  END AS df_bucket,
  COUNT(*) AS num_tokens,
  ARRAY_AGG(token ORDER BY df DESC) FILTER (WHERE df >= 200) AS sample_tokens
FROM token_frequency
GROUP BY df_bucket
ORDER BY MIN(df) DESC;
```

Pick the ceiling at the first bucket boundary below which the
tokens stop looking like real brand signal. Expected shape:

- `>=10000`: boilerplate ("pictures", "films", "studios")
  already in stoplist.
- `2000-9999`: stoplist residuals — consider extending stoplist.
- `500-1999`: borderline. Decision lives here.
- `200-499`: clearly brand tokens (disney, mgm, pixar, ghibli).
- `<200`: long-tail brand tokens and singletons.

Starting recommendation: **ceiling = 500**. Validate by eyeballing
`200-499` and `500-1999` bucket contents.

## Edge Cases and How They're Handled

### Umbrella conglomerate queries (D3, D4, D8)

Handled entirely by the brand path. `brand_id = DISNEY` retrieves
movies stamped with the Disney brand at ingest, respecting
time-bounded ownership via `brand_member_company`. Lucasfilm
movies before 2012 do not appear; movies after 2012 do. Legacy
`Twentieth Century Fox` and post-rename `20th Century Studios`
both map to the appropriate brand(s) with correct year bounds.

### Acronym with non-embedded long form (D1)

MGM queries: LLM emits `brand_id: MGM`. The brand path retrieves
all MGM-stamped movies regardless of whether the string variant
embeds the acronym (`Metro-Goldwyn-Mayer (MGM)`) or doesn't
(`Metro-Goldwyn-Mayer Cartoon Studios`) — both company IDs are
members of the MGM brand.

If the LLM instead emits `freeform_names: ["mgm",
"metro-goldwyn-mayer"]` (e.g., if MGM isn't in the registry
yet), union captures both variant classes: the first name covers
MGM-acronym variants via the `mgm` token; the second covers
long-form-only variants via the hyphen-split `{metro, goldwyn,
mayer}` tokens.

### Orphan acronym with no long form (D1)

NHK queries: LLM emits `freeform_names: ["nhk"]` only. No
semantic translation — "japan broadcasting corporation" would
return zero hits because IMDB doesn't use that form. Prompt
explicitly instructs against translation.

### User-common acronym not in data (D1)

"WB" queries: LLM emits `freeform_names: ["warner bros",
"warner brothers"]` (nickname translation via prompt guidance,
not stored alias). The LLM's prompt includes a short list of
these common colloquialisms; the code never sees "WB" itself.

### Specificity preservation (D7)

"Walt Disney Animation" queries: LLM emits `brand_id: null,
freeform_names: ["walt disney animation studios", "walt disney
animation"]`. Token intersection narrows to animation
subsidiaries only. Walt Disney Pictures and Disney Channel are
correctly excluded.

### Ambiguous common-word tokens (D5)

"Universal" queries: LLM emits `brand_id: UNIVERSAL`. Brand path
returns movies stamped with the Universal brand via
`brand_member_company`. Unrelated `universal`-containing strings
(Geneon Universal, Universal Music) aren't in that brand's
member list, so they don't leak through.

### Numeric ↔ word forms (D2)

"20th Century Fox": normalizer rewrites to `twentieth century
fox` at both ingest and query time. Matches the dominant
`Twentieth Century Fox` tags directly. The brand path's
`brand_member_company` covers both numeric and word variants
because both strings normalize identically and map to the same
`production_company_id`.

### Long-tail specific name (D1 long tail)

"Villealfa Filmproductions": `brand_id: null, freeform_names:
["villealfa"]`. Token has DF=1; maps to exactly one
`production_company` row. No brand membership applies.

### LLM sets brand_id when user wanted specific (D10 recovery)

Prompt explicitly tells the LLM: "Set brand_id only for umbrella
queries. For specific sub-labels, leave brand_id null." The
`thinking` field forces the model to reason about scope before
deciding. If the LLM still misfires (brand_id set when user
wanted narrow), the result set is over-broad; the other query
channels (vector, metadata) narrow the overall search.
Browsing UX absorbs the imprecision.

### Institutional / non-studio entries (D6, D9)

`Province of British Columbia Film Incentive BC` has tokens
`{province, british, columbia, film, incentive}` after stoplist.
Not in any brand's member list. Only retrievable if a user
explicitly names enough of its tokens — which real users don't.
Token-index noise, tolerated by the browsing UX.

### New studios after initial setup

Re-run Stages A–D monthly (cheap; no LLM cost). New
`production_company` rows populate; DF and token index refresh.
Stage E re-stamps affected movies only when the brand registry
changes.

### Brand registry updates

When an edge is added (e.g., a new acquisition), re-run Stage E
for affected movies only. No other pipeline changes needed.

### Empty post-prefilter name

If a `freeform_name` contains only boilerplate (`"the studios"`),
the prefilter strips everything and that name contributes
nothing. If all three names collapse to empty, the studio
channel returns empty.

## Worked Examples

### Example 1 — "disney animated classics"

```
LLM output:
  thinking: "User asked for Disney movies broadly, a parent brand
             in the closed registry. Umbrella scope."
  brand_id: DISNEY
  freeform_names: []
Brand path: movie.brand_ids && ARRAY[DISNEY]
Result: all movies stamped with DISNEY at release year —
        Walt Disney Pictures titles, plus Pixar post-2006,
        Marvel Studios post-2009, Lucasfilm post-2012, Touchstone
        during its Disney era, etc.
```

Vector and metadata channels independently handle "animated
classics" narrowing.

### Example 2 — "MGM Studios musicals"

```
LLM output:
  thinking: "MGM is a closed-registry brand. Umbrella scope."
  brand_id: MGM
  freeform_names: []
Brand path: movie.brand_ids && ARRAY[MGM]
Result: all MGM-stamped movies including Metro-Goldwyn-Mayer
        Cartoon Studios and British Studios variants.
```

### Example 3 — "walt disney animation"

```
LLM output:
  thinking: "User named a specific sub-label, not the full Disney
             umbrella. Use freeform_names to narrow."
  brand_id: null
  freeform_names: ["walt disney animation studios",
                   "walt disney animation",
                   "walt disney feature animation"]
Freeform path (intersection within each name):
  "walt disney animation studios" → tokens {walt, disney,
     animation} (studios is boilerplate)
     ∩ → {Walt Disney Animation Studios,
          Walt Disney Feature Animation}
  "walt disney animation" → same result
  "walt disney feature animation" → {Walt Disney Feature Animation}
Union across names: {Walt Disney Animation Studios,
                     Walt Disney Feature Animation}
```

Specificity preserved — user said "animation", user gets
animation subsidiaries only. Walt Disney Pictures and Walt
Disney Productions correctly excluded.

### Example 4 — "A24 indies"

```
LLM output:
  thinking: "A24 is a closed-registry brand, standalone studio."
  brand_id: A24
  freeform_names: []
Brand path: movie.brand_ids && ARRAY[A24]
```

### Example 5 — "Fox movies"

```
LLM output:
  thinking: "User likely wants both the legacy Fox catalog and
             the post-rename 20th Century Studios catalog.
             Closed-registry brand covers both."
  brand_id: TWENTIETH_CENTURY_FOX
  freeform_names: []
Brand path: returns both Twentieth Century Fox-stamped legacy
            movies and 20th Century Studios-stamped post-2020
            movies, via multiple member_company rows on the
            same brand.
```

### Example 6 — "Villealfa Filmproductions" (long tail)

```
LLM output:
  thinking: "Long-tail Finnish studio, not in the closed
             registry."
  brand_id: null
  freeform_names: ["villealfa filmproductions", "villealfa"]
Freeform path:
  "villealfa filmproductions" → {villealfa} (filmproductions
     in stoplist) → posting(villealfa) → {Villealfa Filmproductions}
  "villealfa" → same
Union: {Villealfa Filmproductions}
```

### Example 7 — "NHK documentaries"

```
LLM output:
  thinking: "NHK is a Japanese broadcaster. Only name appearing
             in IMDB is the acronym itself. No translation."
  brand_id: null
  freeform_names: ["nhk"]
Freeform path:
  posting(nhk) → {NHK, NHK Enterprises, NHK Worldwide, ...}
```

### Example 8 — Time-bounded ownership (Star Wars + Disney)

If a combined query names both the original Star Wars and
Disney:

```
LLM output:
  thinking: "User asked for Disney. Closed-registry brand."
  brand_id: DISNEY
  freeform_names: []
Brand path: Star Wars 1977 NOT in result set — Lucasfilm's
            DISNEY membership starts 2012 via brand_member_company
            (start_year=2012), and 1977 is outside that range.
            Star Wars: The Force Awakens 2015 IS in the result.
```

## Open Decisions

1. **DF ceiling.** Pin the empirical value after running the
   bucket analysis in Stage C. Starting guess 500.
2. **Brand registry curation.** Tier list pinned in
   [production_company_tiers.md](production_company_tiers.md)
   (24 Tier 1 MVP brands, 14 Tier 2 phase-2, 10 Tier 3
   enthusiast — ~48 total, matches the ~50 target). Remaining
   work: seed exact `start_year` / `end_year` values from
   Wikidata's `owned by` qualifiers and confirm rename scope per
   Open Decision #7.
3. **Brand granularity.** Pixar, Marvel Studios, Lucasfilm,
   Searchlight as their own brand_ids (so narrow queries work)
   AND as time-bounded members of Disney (so umbrella queries
   work). Overlapping membership is allowed and necessary.
4. **Boilerplate stoplist evolution.** Extend empirically based
   on Stage C bucket review, especially for non-English
   residuals (`cinéma`, `filmproduktion`, `produksjonen`, etc.).
5. **Re-run cadence.** Monthly Stages A–D; Stage E only on
   brand registry changes or new movie ingests.
6. **Exclusion handling.** "Disney movies not Disney Channel"
   stays in the exclusion-lexical layer of the query-
   understanding DAG, not the studio resolver.
7. **Fox rename scope.** Confirm during curation whether
   `Twentieth Century Fox` (pre-2020) and `20th Century Studios`
   (post-2020) should be one brand with two membership rows or
   two separate brands with the Disney edge only on the latter.

## Brand Registry — MVP Tier List

Full rationale, selection criteria, overlap relationships, and
streamer disambiguation rules live in
[production_company_tiers.md](production_company_tiers.md). The
condensed list below is the commitment.

### Tier 1 — MVP must-haves (24 brands)

`DISNEY`, `WALT_DISNEY_ANIMATION`, `PIXAR`, `MARVEL_STUDIOS`,
`LUCASFILM`, `WARNER_BROS`, `NEW_LINE_CINEMA`, `DC`, `UNIVERSAL`,
`FOCUS_FEATURES`, `PARAMOUNT`, `SONY`, `COLUMBIA`,
`TWENTIETH_CENTURY`, `SEARCHLIGHT`, `DREAMWORKS_ANIMATION`,
`ILLUMINATION`, `MGM`, `LIONSGATE`, `A24`, `NEON`, `BLUMHOUSE`,
`STUDIO_GHIBLI`, `NETFLIX`.

### Tier 2 — Phase-2 adds (14 brands)

`SONY_PICTURES_CLASSICS`, `SONY_PICTURES_ANIMATION`, `TRISTAR`,
`SCREEN_GEMS`, `TOUCHSTONE`, `HBO_FILMS`, `LAIKA`, `MIRAMAX`,
`AARDMAN`, `WORKING_TITLE`, `AMBLIN`, `UNITED_ARTISTS`,
`AMAZON_MGM`, `APPLE_STUDIOS`.

### Tier 3 — Post-launch enthusiast (10 brands)

`LEGENDARY`, `ANNAPURNA`, `SKYDANCE`, `BAD_ROBOT`, `PLAN_B`,
`TOHO`, `STUDIOCANAL`, `CJ_ENTERTAINMENT`, `ORION`, `CASTLE_ROCK`.

### Excluded on purpose

- **Streamers-as-platforms** (`NETFLIX-as-platform`, `APPLE_TV+`,
  `PRIME_VIDEO`, `HULU`, `MAX`, `PEACOCK`, `DISNEY+`) — handled by
  the `watch_providers` metadata path, not the studio resolver.
  Note `NETFLIX`, `AMAZON_MGM`, and `APPLE_STUDIOS` are registered
  as producer brands but the prompt rules the studio channel out
  when the user's intent is streaming availability (see prompt
  contract above).
- **Pure financiers / coproducers** (`Syncopy`, `Village
  Roadshow`, `Regency`, `Participant`, `TSG`, `IAC Films`,
  `Perfect World`, `Entertainment One`) — freeform token path
  covers the rare case.
- **Deep-cult back catalog** (`Cannon`, `Carolco`, `Troma`,
  `Hammer`, `Janus`, `IFC`, `GKIDS`, `RKO`) — promote case-by-case
  from query-log evidence post-launch.

## Relationship to Existing System

- **Replaces** the exact-match studio lookup path currently in
  `lex.lexical_dictionary` usage for studios.
- **Extends** the existing posting-list retrieval — once the
  resolver produces a set of canonical strings, the downstream
  movie posting list code is unchanged.
- **Aligns with** the conventions in `docs/conventions.md`:
  - "Instruct both sides to use the most common form" — LLM
    emits common surface forms; code normalizes and resolves.
  - "GIN arrays for enums, posting tables for text entities" —
    `movie.brand_ids INT[]` is a GIN array; token index is a
    posting table.
  - "Don't ask LLMs to do what code can derive" — ownership
    structure lives in `brand_member_company`, not the LLM.
  - "Closed enum for freeform axes" — matches the
    award-ceremony and award-category precedents.
- **Supersedes** the open question at the end of
  [finalized_search_proposal.md:3118](finalized_search_proposal.md#L3118)
  ("Studio name matching brittleness — currently exact-only
  after normalization; may need LIKE substring or alias table
  if too many misses in practice").

---

# Related Endpoints With the Same Class of Problem

The studio resolver above is one instance of a broader pattern:
freeform natural-language terms must map to canonical strings
stored in the DB, and naive exact-match after normalization is
brittle. Two other endpoints in stage 3 carry the same structural
risk: **franchise** (lineage / shared universe / subgroup names)
and **award names** (prize names like "Oscar", "Palme d'Or").
This section inventories each — how data gets in, how the query-
time LLM generates matching terms, how the two sides are joined,
and where the joins can silently fail — so we can plan analogous
fixes.

## Franchise Endpoint

### 1. Data Ingestion and Storage

Franchise data lives in `public.movie_franchise_metadata`, one
row per movie ([01_create_postgres_tables.sql:114](db/init/01_create_postgres_tables.sql#L114)):

```
lineage                TEXT         -- narrowest franchise line
shared_universe        TEXT         -- broader cosmos / parent
recognized_subgroups   TEXT[]       -- named phases / sagas / eras
lineage_position       SMALLINT     -- enum: sequel / prequel / remake / reboot
launched_subgroup      BOOLEAN
is_spinoff             BOOLEAN
is_crossover           BOOLEAN
launched_franchise     BOOLEAN
```

The three freeform text fields (`lineage`, `shared_universe`,
`recognized_subgroups`) are fully open-vocabulary — no closed
enum, no lookup dictionary, no alias table. Every string that
ends up there was written once by an LLM looking at a single
movie and deciding what to type.

Population happens in the Stage 6 metadata generation pipeline
([generators/franchise.py](movie_ingestion/metadata_generation/generators/franchise.py))
using gpt-5.4-mini with `reasoning_effort=low`, driven by the
system prompt in [prompts/franchise.py](movie_ingestion/metadata_generation/prompts/franchise.py).
Generated values are passed through `normalize_string` (lowercase,
diacritic fold, punctuation strip) before being stored, so the
stored column is already normalized.

The ingest-side prompt spells out canonical-naming rules: most
common well-known form, lowercase, spell digits as words, expand
"&" to "and", expand abbreviations only when the expanded form is
also in wide use (MCU → marvel cinematic universe; LOTR → the
lord of the rings; monsterverse stays monsterverse; x-men stays
x-men), drop first names from director-era labels when the
surname alone is common ("peter jackson's lord of the rings
trilogy" → "jackson lotr trilogy").

### 2. How the LLM Generates Data in Stage 3

The stage-3 franchise translator
([franchise_query_generation.py](search_v2/stage_3/franchise_query_generation.py))
emits a `FranchiseQuerySpec` with up to three entries each in
`lineage_or_universe_names` and `recognized_subgroups`, plus
structural flags and `lineage_position`. The LLM is told to use
the **same canonical-naming rules** as the ingest-side generator,
with the rule set duplicated in the prompt.

The guidance: emit 1 primary form for the common case; emit 2-3
alternates only when there are genuinely different canonical
forms in wide use (e.g., "marvel cinematic universe" and "marvel";
"the lord of the rings" and "middle-earth"). Do not pad with
spelling or casing variants.

### 3. Joining LLM Output to Stored Data

[franchise_query_execution.py:43-61](search_v2/stage_3/franchise_query_execution.py#L43-L61)
runs `normalize_string` on each query-side variation, then hands
the list to [fetch_franchise_movie_ids](db/postgres.py#L1897),
which does strict string equality:

```
(lineage = ANY(variants) OR shared_universe = ANY(variants))
AND EXISTS (
  SELECT 1 FROM unnest(recognized_subgroups) sg
  WHERE sg = ANY(subgroup_variants)
)
```

Both sides are pre-normalized — no LOWER(), no trigram, no LIKE,
no token intersection. One-character mismatch = zero hits.

### 4. How This Introduces Mismatches

The core problem: **two independent LLMs, run months apart on
different movies, are expected to converge on identical canonical
strings for the same concept.** The rules are the same in both
prompts, but agreement is not guaranteed.

Specific failure modes:

- **Cross-model drift.** Ingest uses gpt-5.4-mini; query-time uses
  Kimi (or whichever provider is routed). Different families make
  different "common form" judgments. If ingest wrote
  `"spider-man"` and query emits `"spiderman"`, they normalize to
  different strings and never match.
- **Within-ingest drift.** The ingest LLM sees one movie at a
  time. There is no shared dictionary across movies, so different
  movies in the same franchise can end up with subtly different
  lineage strings — `"the lord of the rings"` on one movie,
  `"lord of the rings"` on another.
- **Subgroup fragmentation.** Subgroups are the most variable —
  "phase one" vs "phase 1", "infinity saga" vs "marvel phase one",
  "snyderverse" vs "snyder verse" vs "snyder-verse". Even with the
  prompt telling both sides the same rules, a single unexpected
  tokenization choice on one movie silently removes it from the
  subgroup's result set.
- **Three-guess ceiling.** Query-time caps
  `lineage_or_universe_names` at three entries. If the stored form
  isn't one of the three guesses the query-time LLM makes, the
  whole franchise axis misses regardless of how obviously the
  query matches the concept.
- **Rename and abbreviation drift.** "DCEU" / "dc extended
  universe" / "dc universe" / "snyderverse" — the prompt handles
  a handful explicitly, but the space of aliased shorthands is
  open-ended and grows over time. Every new alias is a potential
  silent miss until someone notices and adds it to both prompts.
- **No fallback layer.** Unlike studios, franchise has no token
  intersection fallback. If exact match misses, the whole endpoint
  contributes zero signal.

### 5. Prior Solutions in the Stage-3 Pipeline

Two precedents in this codebase show strategies that *do* work
for freeform-to-canonical mapping, both within the award endpoint
itself:

- **Award categories** used to be free-text exact-match. They are
  now a closed 3-level taxonomy (`CategoryTag` leaf → mid → group)
  stored as `category_tag_ids INT[]` with ancestor expansion at
  ingest ([01_create_postgres_tables.sql:88-93](db/init/01_create_postgres_tables.sql#L88-L93)).
  Any ingest-side raw category string gets consolidated into the
  closed taxonomy via `consolidate_award_categories.py`. Query
  side emits an enum value; retrieval is GIN `&&` overlap. The
  LLM can choose at any specificity and the row's stored ancestor
  list does the rest.
- **Award ceremonies** are a closed `AwardCeremony` enum with
  numeric IDs. The prompt includes a programmatically-rendered
  table ([render_ceremony_mappings_for_prompt](schemas/award_surface_forms.py#L86))
  mapping natural-language event names to exact stored enum
  values. The LLM cannot invent a ceremony — structured output
  restricts it to the closed set.

Both replaced exact string matching with closed-enum ID matching.
The pattern that emerges: **any freeform axis that both sides
must agree on is a canonicalization liability**; move it to a
closed enum or an auto-built index the moment the open-vocabulary
space starts producing misses.

## Award Name Endpoint

### 1. Data Ingestion and Storage

`public.movie_awards.award_name TEXT`
([01_create_postgres_tables.sql:86](db/init/01_create_postgres_tables.sql#L86))
stores the **raw IMDB surface form** of each prize — "Oscar",
"Palme d'Or", "Golden Lion", "BAFTA Film Award", plus thousands
of festival-specific prize names for in-scope ceremonies.

Ingestion is direct extraction from IMDB's GraphQL response
([parsers.py:290-293](movie_ingestion/imdb_scraping/parsers.py#L290-L293)):

```
award_name = _safe_get(node, ["award", "text"])
...
award_name = award_name.strip()
```

That is the entire transformation: `.strip()`. No case fold, no
diacritic strip, no apostrophe normalization, no consolidation.
Whatever IMDB returns is what gets stored, including any
inconsistencies IMDB itself has across prizes or over time.

### 2. How the LLM Generates Data in Stage 3

The stage-3 award translator
([award_query_generation.py](search_v2/stage_3/award_query_generation.py))
emits an `AwardQuerySpec` whose `award_names` field is a list of
strings. The prompt renders a canonical surface-form registry
([schemas/award_surface_forms.py](schemas/award_surface_forms.py))
covering **12 ceremonies**: "Oscar", "Golden Globe", "BAFTA Film
Award", "Palme d'Or", "Grand Jury Prize", "Jury Prize", "FIPRESCI
Prize", "Golden Lion", "Silver Lion", "Golden Berlin Bear",
"Silver Berlin Bear", "Teddy", "Actor" (SAG), "Critics Choice
Award", "Razzie Award", "Independent Spirit Award", "Gotham
Independent Film Award".

The prompt explicitly tells the LLM: **"The table is not a closed
vocabulary. Do NOT restrict output to table entries."** For
off-table prizes, the LLM is told to "use your knowledge of IMDB
nomenclature for the relevant ceremony; do not approximate to a
similar-looking table entry."

### 3. Joining LLM Output to Stored Data

[award_query_execution.py:62-83](search_v2/stage_3/award_query_execution.py#L62-L83)
deduplicates the list but **deliberately does not normalize**:

```
# Does NOT normalize — stored values keep their raw IMDB surface
# form and the comparison must be exact. No case-folding, no
# diacritic stripping, no whitespace trimming: any of those could
# silently mask a real stored/query mismatch.
```

The downstream SQL filter uses raw string equality
(`award_name = ANY(%s)`) against the raw IMDB surface form in the
column. The comment is accurate about the risk — case-folding
here would silently coerce mismatched pairs into matching — but
the consequence is that the query-time LLM must reproduce IMDB's
exact surface form, including punctuation and capitalization, or
the row is invisible.

### 4. How This Introduces Mismatches

The award-name axis is the only remaining free-text lookup in the
award endpoint (ceremonies and categories have already been
closed-enumerated). Its failure modes:

- **Surface-form dependence on IMDB quirks.** "Palme d'Or" with a
  curly apostrophe (U+2019) vs a straight one (U+0027) are
  different strings. IMDB's own data isn't guaranteed to be
  consistent here, and neither is LLM output.
- **Registry is small; space of prizes is large.** The 12-entry
  registry covers flagship prizes for 12 ceremonies. Every other
  festival prize — Sundance's per-section awards, regional
  festival prizes, niche awards — relies on the LLM recalling the
  exact IMDB surface form from memory. Small LLMs (Kimi) are not
  reliable at this.
- **Forced-choice under structured output (related to the earlier
  audio_language discussion).** The schema accepts any string, so
  structured output doesn't catch misspellings; the LLM freely
  emits plausible-looking but non-stored forms like
  `"BAFTA Award"` instead of `"BAFTA Film Award"`.
- **IMDB surface-form inconsistency.** The same ceremony's prize
  name may differ across years if IMDB renamed it — `"BAFTA"` vs
  `"BAFTA Film Award"` vs `"BAFTA Award"` — and we store whatever
  IMDB returned on each scrape. No consolidation pass exists, so
  a single prize concept may be split across multiple
  `award_name` strings in our DB. Query-side must guess which one
  the ingest pipeline actually stored on the movies it's looking
  for.
- **No fallback.** Exact match misses → endpoint contributes
  zero. No trigram, no token intersection, no alias table.

The mirror of the franchise problem: ingest writes whatever the
source returned, query-time writes whatever the LLM decides
"looks canonical" — agreement between the two is by luck.

### 5. Prior Solutions in the Stage-3 Pipeline

Same precedents as franchise, and this is the endpoint where the
precedents are most directly applicable — because the other two
axes of this very table have already been solved this way:

- **`category`** (free-text TEXT column, now superseded by
  `category_tag_ids INT[]`) — consolidated to a closed
  `CategoryTag` enum at ingest. Query side uses the enum and
  retrieval is GIN overlap on integer IDs. This is the direct
  template for what `award_name` should become.
- **`ceremony`** (free-text at IMDB, stored as `ceremony_id
  SMALLINT`) — mapped through the `AwardCeremony` enum with a
  registry-driven prompt table. Query side cannot produce an
  invalid ceremony because the enum constrains structured output.

The missing piece is the same treatment for `award_name`:
consolidate IMDB surface forms into a canonical registry (closed
or semi-closed), store a `prize_id` alongside or instead of the
raw string, and drive the query-time prompt from the same
registry. Exactly analogous to how the studio resolver above
replaces per-movie freeform strings with a brand registry plus a
token-index fallback.

## Summary Comparison

| Endpoint | Axis | Current state | Closed-enum equivalent? |
|----------|------|---------------|--------------------------|
| Studio | production_company | Brand registry + time-bounded membership + token index (this doc) | ✓ solved |
| Franchise | lineage / shared_universe / subgroups | Freeform text, exact match after normalize_string | None — still at risk |
| Awards | ceremony | Closed enum (AwardCeremony), registry-driven prompt | ✓ solved |
| Awards | category | Closed 3-level taxonomy (CategoryTag), GIN overlap | ✓ solved |
| Awards | award_name | Raw IMDB surface form, exact match, no normalize | None — still at risk |

Franchise and award_name are the two remaining stage-3 axes where
the query side is expected to guess a string the ingest side
wrote. Both warrant the same treatment that studios, award
categories, and award ceremonies have already received: move the
canonical form into data, not into two prompts' independent best
guesses.

---

# Studio Matching — Full Problem Dimensions

The resolver design above was drafted after empirically surveying
IMDB's `production_companies` field. The dimensions below
enumerate every mismatch class observed in the data. The design
references this list for coverage; future work can reference
it to evaluate new edge cases.

## Working assumption going in

We index every entry in `imdb_data.production_companies` as-is
(no role filtering, no attribute-weighting). Over-fetching is
acceptable because the app is a browsing UI, not a single-result
retrieval system. The problem is therefore narrowed to
**query-to-stored-string matching**, not to curating which
companies "count" as producers.

## D1 — Short-form ↔ long-form acronym asymmetry

Three sub-cases, all real in the data:

- **Acronym embedded in long form** (most common):
  `Metro-Goldwyn-Mayer (MGM)`, `British Broadcasting Corporation (BBC)`,
  `Zweites Deutsches Fernsehen (ZDF)`, `Home Box Office (HBO)`,
  `Nippon Television Network (NTV)`. The acronym is a token in
  the string, so acronym-based token match works.
- **Only the long form exists, no acronym anywhere in data**:
  `Metro-Goldwyn-Mayer Cartoon Studios` (276 tags),
  `Metro-Goldwyn-Mayer British Studios` (39). User types "MGM" →
  misses ~11% of the MGM catalog without brand-path fallback.
- **Only the acronym exists, no long form anywhere**: `NHK` has
  242 standalone tags; no `Nippon Hoso Kyokai` string exists. A
  user typing the long form gets zero hits.
- **User-common acronym isn't IMDB's convention at all**: `WB`
  has zero standalone tags. `Warner Bros.` is what IMDB uses. A
  user typing "WB" finds nothing useful.

## D2 — Orthographic / spacing / script-form variants

| Pattern | Data | Variance |
|---|---|---|
| Numeric vs worded | `20th Century Fox` 19 tags vs `Twentieth Century Fox` 1,361 tags | Token sets disjoint — `20th` ≠ `twentieth` |
| Spacing | `Lionsgate` 327 tags vs `Lions Gate Entertainment` 21 | `lionsgate` is one token; `lions gate` is two |
| Hyphen | `TriStar` 131 vs `Tri-Star` 76 | Handled by hyphen-splitting in the tokenizer |
| Punctuation | `Canal+`, `Warner Bros.` | Stripped via normalization — OK |
| Camel case | `DreamWorks Pictures` 138, `Dream Works` effectively 0 | IMDB is consistent — one-word form only |
| Apostrophes / possessives | `disney's` 0 tags | IMDB never uses possessives; handled via normalization |

The numeric ↔ word gap (`20th` / `twentieth`) is the biggest real
biter — solved by the ordinal number-to-word normalization rule.

## D3 — Parent conglomerate ↔ subsidiary (token-disjoint)

Subsidiaries whose names share no token with the parent brand.
Users asking for the parent expect the subsidiaries; pure token
match can't bridge.

| Parent | Token-disjoint subsidiaries (tag counts) |
|---|---|
| **Disney** | Pixar (132), Marvel Studios (82), Lucasfilm (94), Touchstone Pictures (230), Miramax (272), Hollywood Pictures (115), Fox Searchlight (105), Searchlight Pictures (53), 20th Century Fox (1,457), 20th Century Studios (56), Buena Vista variants (75) |
| **Warner** | HBO-family (~820 across 10+ strings), New Line Cinema (367), DC Comics / Entertainment / Films (~160), Castle Rock Entertainment (103), Turner / TNT (~84) |
| **Sony** | Columbia Pictures (2,192), TriStar (131+76), Screen Gems (181), Columbia TriStar Television (51) |
| **NBCUniversal** | Focus Features (124), Illumination Entertainment (59), Working Title Films (156), DreamWorks Animation (120) |
| **Paramount** | Nickelodeon-family (258 across 7 strings), MTV Films (50) |
| **Amazon** | Metro-Goldwyn-Mayer (MGM) (2,267 legacy), Amazon MGM Studios (96 post-acquisition) |

Token-only Warner coverage misses ~28% of the conglomerate
catalog; Disney misses ~34%. Solved by the brand registry +
time-bounded membership.

## D4 — Historical renames (the canonical form changed)

| Old | New | Ambiguity |
|---|---|---|
| 20th Century Fox | 20th Century Studios (2020) | `fox` token absent from new form (56 tags orphaned) |
| Buena Vista Pictures Distribution | Walt Disney Studios Motion Pictures (2007) | Zero token overlap |
| Orion Pictures (1978–99) | Orion Pictures (2014–, MGM revival) | Same name, different eras, different owners |
| Carolco Pictures → extinct 1996 | — | Query returns nothing new; fine |
| MGM (independent) → Amazon MGM Studios (2022) | Same brand token survives | Safe |

Solved by the same mechanism as D3 — rename edges become
additional `brand_member_company` rows on the same brand.

## D5 — Ambiguous tokens shared across unrelated brands

The same token names multiple, corporately unrelated entities:

| Token | Distinct strings | Brands implied |
|---|---|---|
| `universal` | 145 | UNIVERSAL-film (Comcast) + Geneon Universal (Japan, unrelated) + Universal-music + TV/cable entities |
| `columbia` | 69 | Columbia Pictures (Sony) + **`Province of British Columbia Film Incentive BC` (88 tags) + `...Production Services Tax Credit` (75)** + Columbia TriStar TV + others |
| `fox` | 167 | 20th Century Fox family + Fox Searchlight + Sidney Fox Productions + unrelated Fox entities |
| `focus` | 77 | Focus Features (Universal) + Prime Focus (35), Prime Focus World (21), Learning in Focus (18), Hocus Focus Productions (11) |
| `apple` | 75 | Apple Studios / Apple Original Films / Apple Film Productions + Green Apple (17), Black Apple Media (7), etc. |
| `neon` | 26 | NEON the A24-era distributor + Neon Heart (23), Neon Rouge (22), Neon Sheep, etc. |
| `hammer` | 15 | Hammer Films (horror) + Velvet Hammer, Sledgehammer, etc. |
| `orion` | 19 | Orion Pictures (1978–99 + 2014–) + unrelated Orion entities |

Deterministic routing for the worst offenders comes from the
brand registry: `brand_id = UNIVERSAL` only returns member
companies, not arbitrary `universal`-containing strings.

## D6 — Geographic / institutional collision with brand tokens

Government and institutional entries in IMDB's production_companies
that share tokens with actual brands:

- `Province of British Columbia Film Incentive BC` (88) —
  collides with "Columbia Pictures"
- `Province of British Columbia Production Services Tax Credit`
  (75) — same
- `Filmuniversität Babelsberg Konrad Wolf` (92) — film school,
  collides with "Studio Babelsberg"
- `National Film Board of Canada` — collides with any
  "national" query
- `Finnish Film Foundation`, `Filmförderungsanstalt (FFA)`,
  `BFI`, `ICAA`, `CNC` — national/regional funders that may
  legitimately be queried OR be noise depending on user intent

Tolerated — browsing UX absorbs the noise.

## D7 — Same-brand different-function units

User typing a brand name expects the movie catalog, but IMDB
strings differentiate sub-units:

- `Walt Disney Pictures` / `...Animation Studios` / `...Television` /
  `...Feature Animation` / `Disney Channel` / `Disney+` — broad
  "Disney" query should hit all; narrow "Walt Disney Animation"
  should hit only the animation subsidiary
- `BBC Film` (435) / `BBC Studios` (136) / `BBC Scotland` (81) /
  `BBC Wales` (55) / `BBC Northern Ireland` (45) / `BBC Worldwide`
  (46) / `BBC Storyville` (41) / `BBC Bristol` (30)
- `HBO Films` (143) / `HBO Documentary Films` (214) / `HBO Pictures`
  (50) / `HBO Europe` (39) / `HBO Max` (33) / `HBO Sports` (32)
- `Warner Bros.` (2,676) / `...Cartoon Studios` (512) /
  `...Animation` (314) / `...Television` (243) /
  `...Film Productions Germany` (53)

Multi-token intersection already handles narrow specificity
(`walt disney animation` narrows correctly). Umbrella queries go
through the brand path.

## D8 — Time-bounded ownership (the brand graph is not static)

Subsidiary membership changes over time:

- **Miramax**: Disney-owned 1993–2010, then sold to Qatari
  owners.
- **Marvel Studios**: independent 1996–2009, Disney-owned after.
- **Lucasfilm**: independent 1971–2012, Disney-owned after.
- **20th Century Fox**: independent until 2019, Disney after
  (renamed 20th Century Studios in 2020).
- **DreamWorks Animation**: independent 2004–2016, Paramount
  briefly 2006–08, Universal after 2016.
- **MGM**: independent until 2022, Amazon after.
- **New Line Cinema**: independent until 1994, Turner/Time
  Warner after.

Solved by `brand_member_company.start_year` / `end_year` bounds,
applied at ingest when stamping `movie.brand_ids`. Star Wars
1977 is not a Disney movie; Force Awakens 2015 is.

## D9 — Institutional non-studio entries in production_companies

The list includes entities that aren't studios in the user's
sense but live in the same column:

- National/regional film funds: `BFI`, `FFA`, `CNC`, `NFB`,
  `Finnish Film Foundation`, `Filmförderungsanstalt`,
  `Filmboard Berlin-Brandenburg`
- Public broadcasters as co-financiers: `ZDF` (2,739 tags),
  `BBC` (3,504), `ARTE`, `Canal+` (4,555), `NHK`
- Tax-incentive programs: `Québec Production Services Tax
  Credit`, `Province of British Columbia...`, `The South
  Australian Film Corporation`
- Film universities: `Filmuniversität Babelsberg Konrad Wolf`

Indexed as companies. Queries self-select: `BBC films` reaches
the BBC entries naturally; tax-credit entries go unretrieved
because users don't query them.

## D10 — LLM-emission behavior (the query-side variable)

The LLM decides what form to emit for a given user utterance,
and its choice isn't guaranteed to match IMDB's convention:

- User types "WB" → LLM must translate to `Warner Bros.` forms,
  not emit `WB`.
- User types "Fox" → LLM could emit `Fox`, `Twentieth Century
  Fox`, or `20th Century Studios`.
- User types "Mouse House" → LLM must translate colloquialism to
  `Disney`.
- User types "Pixar" → clean token, no ambiguity.
- User types "MCU" → LLM must expand to `Marvel Studios`.
- User types "HBO movies" → LLM emits `HBO`, which covers the
  subsidiary but should also trigger `brand_id = WARNER`? (Open
  — decided by prompt rules.)

The `thinking` field in `StudioQuerySpec` forces the LLM to
reason about scope and brand applicability before filling
`brand_id` and `freeform_names`. Prompt rules constrain the
emitted forms to what IMDB's data actually contains.

## D11 — Query intent patterns

What users actually ask for, and what each pattern implies:

| Pattern | Example | Expected result | Mechanism |
|---|---|---|---|
| Broad brand | "Disney movies" | All Disney-owned catalog incl. subsidiaries at release year | Brand path |
| Specific sub-label | "Walt Disney Animation", "Pixar" | Only that specific unit | Freeform path (Pixar also has its own brand_id) |
| Nickname / colloquial | "WB films", "Mouse House" | Parent-brand catalog | LLM prompt translates |
| Historical era | "Fox pre-Disney", "RKO noir" | Catalog before corporate change | Brand path with time bounds |
| Era rename | "Fox movies" in 2026 | Legacy Twentieth Century Fox + new 20th Century Studios | Brand path — both under same brand |
| Acronym | "MGM", "BBC", "A24" | Full brand output | Brand path |
| Specific co-label | "Fox Searchlight" | Narrow label only | Freeform path |
| Ambiguous short-word | "Focus", "Neon", "Orion" | The brand, not homonyms | Brand path for majors; freeform accepts some collision |
| Non-English broadcaster | "Canal+", "ZDF" | That broadcaster's films | Brand path if registered; freeform otherwise |
| Conglomerate parent | "everything Disney owns", "Comcast movies" | Sprawling catalog | Brand path |

## Summary of dimensions

1. **D1 Short-form / long-form asymmetry** — solved by brand
   path (umbrella) and multi-name freeform path (specific).
2. **D2 Orthographic variants** — numeric-vs-word solved by
   ordinal normalization; others by existing normalization.
3. **D3 Conglomerate bridging** — solved by brand registry +
   `brand_member_company`.
4. **D4 Historical renames** — solved by multiple
   `brand_member_company` rows on one brand.
5. **D5 Ambiguous tokens** — deterministic routing via brand
   path for registry majors; long-tail collisions tolerated.
6. **D6 Geographic/institutional collision** — tolerated.
7. **D7 Same-brand sub-unit specificity** — solved by token
   intersection within a freeform_name.
8. **D8 Time-bounded ownership** — solved by ingest-time
   stamping with `start_year` / `end_year`.
9. **D9 Institutional entries** — indexed; queries self-select.
10. **D10 LLM emission variance** — constrained by schema with
    `thinking` field and prompt rules.
11. **D11 Query intent spread** — covered by the two-path
    architecture.

## Scope coverage summary

**Solved:**
- D1 via brand path + multi-name freeform
- D2 numeric↔word via ordinal normalization
- D3 via brand registry
- D4 via time-bounded member edges
- D5 deterministic routing for majors
- D7 specificity via multi-token intersection
- D8 via ingest-time brand stamping

**Accepted noise:**
- D6 institutional collisions (tolerated by browsing UX)
- D9 institutional entries (indexed, queries self-select)
- D10 LLM variance (constrained but not eliminated)

**Controlled by prompt/schema design:**
- D11 query intent patterns via `thinking` field forcing
  scope reasoning before `brand_id` / `freeform_names` choice
