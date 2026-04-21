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

-- Per-movie stamp (postings table, not an array on the movie row):
lex.inv_production_brand_postings
  brand_id             SMALLINT NOT NULL
  movie_id             BIGINT   NOT NULL
  first_matching_index SMALLINT NOT NULL  -- earliest position the brand appears
  total_brand_count    SMALLINT NOT NULL  -- how many brand-member strings hit
  PRIMARY KEY (brand_id, movie_id)
```

**Implementation note:** earlier drafts of this doc called for an
`INT[]` array column on the movie row. The implemented form is a
postings table with `first_matching_index` and `total_brand_count`
columns so prospective prominence-scoring signals are available, even
though the query-side implementation chose **flat 1.0 scoring** for
brand matches (IMDB ordering is unreliable across non-Anglophone
catalogs — see `.claude/plans/ok-this-all-sounds-elegant-melody.md`
for the full rationale). The columns are kept as future features.

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

**lex.inv_production_brand_postings** — the ingest-time stamp. One
row per (brand, movie) pair. Set at ingest by resolving each
`production_companies` string to a `production_company_id`, looking
up `brand_member_company` rows where the movie's release year falls
within `[start_year, end_year]`, and writing one posting per
matched brand. `first_matching_index` is the earliest position in
the movie's raw `production_companies` list where a brand-member
string appears; `total_brand_count` is how many such strings hit.
Indexed on (brand_id, movie_id) PK plus secondary on movie_id for
reverse lookup.

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
3. Write one row per matched brand into
   `lex.inv_production_brand_postings`, carrying the movie_id,
   brand_id, first_matching_index, and total_brand_count.

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
SELECT movie_id
FROM lex.inv_production_brand_postings
WHERE brand_id = :brand_id
```

Flat 1.0 per matched movie — see the "Scoring" note in the
Implementation Note above.

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
6. **Cardinal number-to-word** for pure-numeric tokens with
   integer value in [0, 99]: `Section 8 Productions` →
   `section eight productions`, `Studio 01` → `studio one`,
   `Unit 9` → `unit nine`. Leading-zero variants are collapsed
   (`01` and `1` both resolve to `one`) by parsing to int before
   lookup. Numbers ≥ 100 and non-pure-numeric tokens (`20mm`,
   `A24`, `se7en`) are left untouched — year-like numbers in
   names (`Fox 2000 Pictures`, `Studio 100`) stay in digit form
   because users never spell them out, and converting them would
   produce unnatural tokens that only hurt recall. Ordinal
   substitution runs first so the digit portion of `20th` is
   consumed before the cardinal rule sees it. Compound cardinals
   use hyphens (`twenty-one`), matching the ordinal convention;
   the tokenizer splits on hyphens so `twenty-one` contributes
   `{twenty-one, twenty, one}` to the token set.

   **Shared rule.** The same ordinal + cardinal (0–99,
   pure-numeric) normalization is used for awards and for
   franchise lineage / shared-universe / subgroup names — see
   those sections' Normalization Rules blocks.

**Tokenization splits on whitespace AND hyphens.** This differs
from any title-side tokenizer that does not hyphen-split — flag
to keep in mind when auditing normalization symmetry.
`Metro-Goldwyn-Mayer` tokenizes to `{metro, goldwyn, mayer}`,
which lets a user typing "metro goldwyn mayer" match
`Metro-Goldwyn-Mayer Cartoon Studios` (a variant that has no
embedded "mgm" token). Bare lone-hyphen tokens that survive
whitespace-splitting from names shaped like `X - Y Productions`
are dropped — a standalone `-` carries no matching signal.

**Out of scope:** Roman numerals (no major studio uses them
in a way that affects retrieval), compound-word splitting
(IMDB is consistent one-word on `DreamWorks`, etc.),
abbreviation expansion beyond ordinals and cardinals 0–99.

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
Brand path: SELECT movie_id FROM lex.inv_production_brand_postings
            WHERE brand_id = DISNEY
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
Brand path: SELECT movie_id FROM lex.inv_production_brand_postings
            WHERE brand_id = MGM
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
Brand path: SELECT movie_id FROM lex.inv_production_brand_postings
            WHERE brand_id = A24
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
    brand memberships use a dedicated posting table
    (`lex.inv_production_brand_postings`); the token index is a
    posting table too.
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

| Endpoint | Axis | Current state | Resolution plan |
|----------|------|---------------|------------------|
| Studio | production_company | Brand registry + time-bounded membership + token index | `Studio Entity Resolution` above |
| Franchise | lineage / shared_universe / subgroups | Freeform text, exact match after `normalize_string` | `Franchise Resolution` below — combined-column entry-id arrays + token index |
| Awards | ceremony | Closed enum (AwardCeremony), registry-driven prompt | ✓ already solved |
| Awards | category | Closed 3-level taxonomy (CategoryTag), GIN overlap | ✓ already solved |
| Awards | award_name | Raw IMDB surface form, exact match, no normalize | `Award Name Resolution` below — symmetric normalization + token index + ceremony scoping |

Franchise and award_name are the two remaining stage-3 axes with
the query-side-guesses-what-ingest-wrote problem. Both are
addressed below by the same token-index pattern the studio
resolver uses, simplified for each endpoint's structure. Neither
uses a registry: award_name leans on the already-closed
ceremony enum for disambiguation, and franchise's open-vocabulary
space is too large to maintain as a closed enum — the ingest-side
canonical-naming rules plus token intersection carry the
retrieval.

---

# Award Name Resolution

## Context

`public.movie_awards.award_name` currently stores the raw IMDB
surface form (`.strip()` only), and stage-3 retrieval uses exact
string equality — see
[award_query_execution.py:62-83](search_v2/stage_3/award_query_execution.py#L62-L83).
This fails every time the query-side LLM emits a surface form that
doesn't match what IMDB stored: straight-vs-curly apostrophes on
`Palme d'Or`, `Critics Week Grand Prize` vs `Critics' Week Grand
Prize`, `BAFTA Award` vs `BAFTA Film Award`, etc. The 12-entry
surface-form registry in
[schemas/award_surface_forms.py](schemas/award_surface_forms.py)
covers flagship prizes only; everything outside it relies on the
LLM reproducing IMDB's exact string from memory.

The fix ports the freeform half of the studio resolver —
symmetric normalization + a token-inverted index + within-name
intersection / across-name union — without the registry /
brand-path overhead. The ceremony axis is already a closed enum
(`AwardCeremony`) and `movie_awards.ceremony_id` is applied as a
row-level filter alongside the entry-id resolution, which kills
cross-festival homonyms ("Grand Jury Prize" at both Cannes and
Sundance) without needing to partition the entry table by
ceremony.

Data shape (empirical):

- 588 distinct `award_name` strings across 12 ceremonies.
- 7 ceremonies have ≤7 distinct names (Oscar, Golden Globe, SAG
  Actor, Razzie, Indie Spirit, Gotham, BAFTA-TV-adjacent) —
  trivial for the LLM to get right.
- 4 festival ceremonies (Cannes, Venice, Berlin, Sundance) have
  130–260 distinct names each — this is where surface-form drift
  lives and where the token index earns its keep.
- 174 singletons in the long tail.

## Design Principles

1. **Symmetric normalization replaces the registry.** Both ingest
   and query run the same `normalize_award_string` +
   `tokenize_award_string` pair, so surface-form drift collapses
   on both sides. Stopword removal diverges — see Principle #5
   — but the string-transform that comes before it is identical.
   The LLM emits the *official* form of the prize, not the exact
   IMDB string.
2. **Ceremony scoping happens at the row level, not the entry
   level.** The entry table is keyed purely on the normalized
   string, so apostrophe/punctuation variants collapse to a
   single entry. `movie_awards.ceremony_id` — already filtered by
   the ceremony channel in the final SQL — handles cross-festival
   homonyms on its own. Partitioning entries by ceremony would
   duplicate data without adding retrieval precision.
3. **Specificity comes from the user, not the LLM.** The prompt
   instructs the LLM to emit the base prize name (`"palme d'or"`)
   NOT overly specific sub-variants (`"palme d'or - best short
   film"`) unless the user asked for the sub-variant. Token
   intersection then greedily matches all sibling variants of a
   prize family, which is the right default for umbrella queries.
4. **No registry, no alias table, no per-ceremony curation.** The
   token index is auto-built from the existing 588 distinct
   `award_name` strings. New prize names picked up on future
   scrapes enter the index on the next ingest.
5. **Stopword droplist at query time only, not ingest.** A short
   closed list of domain-boilerplate words (`award`, `prize`,
   `best`, `the`, `of`, etc. — see the "Stopword Droplist"
   section below) is dropped symmetrically at query time only,
   unlike franchise which drops at both sides. The ingest path
   writes every surviving token to `lex.award_name_token` so the
   token index stays the full empirical corpus — future droplist
   refinements become a config change, not a re-ingest. The
   domain is tiny enough (~600 entries, ~2.2k token rows) that
   the storage cost of keeping the stopwords is negligible.

## Data Model

```sql
CREATE TABLE award_name_entry (
  id          SERIAL PRIMARY KEY,
  normalized  TEXT NOT NULL UNIQUE  -- output of normalize_string
);

CREATE TABLE award_name_token (
  token                TEXT NOT NULL,
  award_name_entry_id  INT  NOT NULL REFERENCES award_name_entry(id),
  PRIMARY KEY (token, award_name_entry_id)
);

-- Btree on token drives the posting-list lookup.
-- Secondary index on entry_id supports reverse lookups / debugging.
CREATE INDEX idx_award_name_token_token ON award_name_token (token);
CREATE INDEX idx_award_name_token_entry ON award_name_token (award_name_entry_id);

-- movie_awards keeps its raw award_name column; a new
-- award_name_entry_id FK is added for deterministic join-back
-- after token resolution.
ALTER TABLE movie_awards
  ADD COLUMN award_name_entry_id INT REFERENCES award_name_entry(id);

CREATE INDEX idx_movie_awards_entry ON movie_awards (award_name_entry_id);
```

**award_name_entry** — one row per distinct normalized string.
Raw IMDB surface forms that collapse to the same normalized form
(straight vs curly apostrophe, `Critics Week` vs `Critics' Week`,
case differences, diacritic differences) merge into a single
entry. The raw form is still available on `movie_awards.award_name`
for display or debugging, but the entry table does not duplicate
it.

Ceremony is **not** part of the entry key. Cross-festival
homonyms are handled by the existing row-level
`movie_awards.ceremony_id` filter, not by partitioning entries.

**award_name_token** — inverted index from token → entry_id. Plain
btree on `token` is sufficient for exact-token equality lookups.

**movie_awards.award_name_entry_id** — foreign key onto the entry
table, written once at ingest. Retrieval runs `token → entry_ids
→ movies` in two indexed steps rather than dragging raw strings
around.

## Ingestion Pipeline

### Stage A — Populate `award_name_entry`

Sweep distinct `award_name` values from `movie_awards`. Apply
`normalize_string` (same rules used at query time) to each.
Upsert one row per **distinct normalized value** into
`award_name_entry`. Multiple raw strings that collapse to the
same normalized form share one entry row — this is the point of
the normalization.

Backfill `award_name_entry_id` on every row in `movie_awards` by
joining on `normalize_string(movie_awards.award_name) =
award_name_entry.normalized`. One-time backfill; subsequent
ingests resolve new rows against the entry table by the same
normalized-string key. Idempotent — re-running adds only the
newly-seen normalized strings.

### Stage B — Tokenize & Write

For every `award_name_entry`:

1. Tokenize `normalized` on whitespace AND hyphens (same rule as
   studio and title).
2. Write every surviving `(token, award_name_entry_id)` pair to
   `lex.award_name_token`. **No stopword filtering at ingest.**
   Stopword removal is a query-side concern — see Design
   Principle #5 above and the Stopword Droplist section below.
   The only ingest-time filter is a lone-hyphen residue drop,
   matching `tokenize_company_string`.

### Stage C — Compute Token Document Frequencies (diagnostic)

Materialize `lex.award_name_token_doc_frequency` — one row per
distinct token with `COUNT(*)` over `lex.award_name_token`. DF is
measured per `award_name_entry`, not per movie.

This view is **diagnostic**, not a filter. It exists so the
query-side stopword droplist can be curated from real data and
revisited as the corpus grows. See the Stopword Droplist section
for the empirically-chosen list and the "Why Not a DF Ceiling"
rationale that comes from inspecting the actual top-DF
distribution.

### Stage D — Index

Btree on `award_name_token.token` (drives posting-list lookups,
exact-token equality only — GIN not needed unless we later add
fuzzy or prefix matching). Secondary btree on
`award_name_token.award_name_entry_id` for reverse lookups /
debugging. Btree on `movie_awards.award_name_entry_id` for the
final movie-id resolution step.

## Query-Time Resolution

The stage-3 award translator emits:

```python
class AwardQuerySpec(BaseModel):
    thinking: str                 # scope reasoning
    award_names: list[str]        # up to 3 official forms
    # ceremony_id, outcome, year handled by existing fields
```

**Prompt contract:**

- Emit the **official base form** of the prize — `"Palme d'Or"`,
  `"Oscar"`, `"Golden Lion"`, `"Grand Jury Prize"`. Lowercase,
  diacritics, and punctuation don't matter — normalization will
  handle them.
- Do NOT emit overly specific sub-variants (`"Palme d'Or - Best
  Short Film"`, `"Silver Berlin Bear - Honorable Mention"`)
  unless the user explicitly asked for the sub-variant. Token
  intersection will sweep sibling variants naturally for umbrella
  queries.
- Emit multiple `award_names` only when the user genuinely named
  different prizes ("Oscar or Golden Globe winners") or when IMDB
  uses meaningfully different forms that don't share tokens after
  stoplist filtering.
- `thinking` field comes first and must commit scope (umbrella
  prize vs specific sub-variant) before `award_names` is filled.

### Step 1 — Normalize and tokenize each name

Apply `normalize_award_string`. Tokenize on whitespace AND
hyphens. Drop tokens matching the stopword droplist (see the
Stopword Droplist section below). If all tokens drop, the name
contributes nothing — the award channel falls through to Step 5.

### Step 2 — Posting-list lookup

For each surviving token, fetch posting list from
`award_name_token`. **Intersect posting lists within the name**
→ `award_name_entry_id` set.

No ceremony filtering happens at the entry level — the entry
table is ceremony-agnostic. Ceremony disambiguation is applied
in Step 4 via the `movie_awards.ceremony_id` filter the award
endpoint already emits.

### Step 3 — Union across names

Across `award_names`, union the per-name entry-id sets.

### Step 4 — Resolve to movies

```sql
SELECT movie_id FROM movie_awards
WHERE award_name_entry_id = ANY(:entry_ids)
  AND ceremony_id         = ANY(:ceremony_ids)      -- if resolved
  AND category_tag_ids   && :category_tag_ids       -- if resolved
  AND outcome_id          = ANY(:outcome_ids)       -- if resolved
  AND year BETWEEN :year_lo AND :year_hi            -- if resolved
```

The entry-id filter plus the existing row-level filters on
`ceremony_id`, `category_tag_ids`, `outcome_id`, and `year` (all
already emitted by the rest of the award endpoint) together
narrow to the correct rows. Cross-festival homonyms like
`Grand Jury Prize` get segregated here, not in Step 2.

### Step 5 — No further fallbacks

If token resolution returns empty, the award channel contributes
nothing. Vector and metadata channels carry the search.

## Stopword Droplist

Applied only at query time (Step 1 above). The ingest path keeps
every token so this list can be revised from the DF view without
re-ingesting — see Design Principle #5.

```
award, awards, prize, prizes,
film, films, best,
a, an, and, for, of, the
```

Revisit cadence: after every ingest sweep, read the top 25 rows
of `lex.award_name_token_doc_frequency`. If a new domain-
boilerplate or English-connective token surfaces, extend the
list; if a word currently on the list turns out to be carrying
real signal in some prize family, pull it off. The list is tiny,
closed, and hand-reviewable — no threshold to tune.

### Why Not a DF Ceiling

The initial design left the stopword-vs-DF-ceiling choice to
post-ingestion. The post-backfill DF distribution (585 entries,
621 distinct tokens) showed why a numeric ceiling fails:

```
token           DF   % of entries
award          339   58%   ← domain boilerplate
special        102   17%   ← DOMAIN-MEANINGFUL (Special Jury Prize)
prize           95   16%   ← domain boilerplate
mention         77   13%   ← DOMAIN-MEANINGFUL (Honorable Mention)
film            59   10%   ← domain boilerplate
jury            49    8%   ← DOMAIN-MEANINGFUL (Grand Jury Prize)
the             45    8%   ← English stopword
of              43    7%   ← English stopword
best            38    7%   ← domain boilerplate
cinema          26    4%   ← DISCRIMINATIVE
golden          24    4%   ← DISCRIMINATIVE (Golden Lion, Golden Bear)
international   24    4%   ← DISCRIMINATIVE
grand           23    4%   ← DISCRIMINATIVE (Grand Jury / Grand Prix)
regard          23    4%   ← DISCRIMINATIVE (Un Certain Regard)
un              21    4%   ← DISCRIMINATIVE (Un Certain Regard)
```

There is no single numeric threshold that cleanly splits signal
from noise:

- Cut above 30 → correctly drops `award`, `special`, `prize`,
  `mention`, `film`, `jury`, `the`, `of`, `best`. But `jury` and
  `mention` are load-bearing. A user typing "Grand Jury Prize"
  then intersects on `{grand}` alone and every "Grand …" prize
  matches.
- Cut above 60 → keeps `jury`, `the`, `of`, `best`. `the` and
  `of` are pure noise that inflate posting-list intersection.
- Cut above 100 → keeps everything except `award` and `special`,
  which defeats the point.

The distribution is **tri-modal and overlapping**: pure English
connectives, domain-meaningful high-DF words, and discriminative
prize signal all coexist in the 20–100 band. The bad tokens are
**semantically bad, not statistically bad** — they need to be
named, not counted.

Notably excluded from the droplist:

- `special` (DF 102) — is a wrapper in "Special Award" but a real
  signal in "Special Jury Prize" (a distinct prize at multiple
  festivals). Keep it and let the ceremony filter narrow.
- `mention` (DF 77) — "Honorable Mention" / "Special Mention"
  appear across many festivals, but a user typing "Special
  Mention" without `mention` keeps no signal at all.
- `jury` (DF 49) — load-bearing across the Cannes / Sundance /
  Venice "Jury Prize" family.

With only ~600 entries and ~2.2k token rows total, a closed hand-
curated droplist is cheap. The materialized DF view stays as the
observation surface for future curation.

## Normalization Rules

Applied symmetrically at ingest and query. **The "deliberately
do not normalize" comment in
[award_query_execution.py:62-83](search_v2/stage_3/award_query_execution.py#L62-L83)
must be removed as part of this change.** The risk that comment
called out (silent coercion of mismatched pairs) is real in
theory, but the current cost — no case-fold, no apostrophe fold,
no diacritic strip — is strictly worse in practice. Symmetric
normalization at both ingest and query is the replacement
invariant.

1. Lowercase.
2. Diacritic-fold (NFKD → strip combining marks) — handles
   `Spéciale`, `d'Or` with combining acute, etc.
3. Strip punctuation (`.`, `,`, `:`, `;`, `'`, `"`, `(`, `)`, `-`,
   `/`, etc.) — folds straight and curly apostrophes to nothing
   (`Palme d'Or` U+0027 vs U+2019 both → `palme dor`), and fixes
   the `Critics Week Grand Prize` / `Critics' Week Grand Prize`
   variant split.
4. Collapse whitespace.
5. Ordinal number-to-word (shared with studio/title) — future-
   proofing against IMDB prize names that acquire ordinals.
6. **Cardinal number-to-word** for pure-numeric tokens with
   integer value in [0, 99] (shared with studio and franchise).
   Same bounds and rationale as the studio rule — see Studio
   Normalization Rules. Applied here so award strings like
   `8th Annual Critics' Week Grand Prize` (ordinal path) and
   hypothetical future prize variants with bare numbers (`Award
   8`) stay symmetric with query-side phrasings.

**Tokenization: whitespace AND hyphens**, same as studio / title.

## Edge Cases

### Flagship prize with sub-variants (Palme d'Or)

Query: "Palme d'Or winners".
```
LLM: award_names = ["palme d'or"]
Normalize → "palme dor"
Tokenize → {palme, dor}
Intersect → entries: {Palme d'Or, Palme d'Or - Best Short Film,
                      Palme d'Or Spéciale}
```
All sibling variants returned — umbrella intent satisfied. If the
user asked "Palme d'Or Best Short Film", LLM emits that narrower
form; tokens `{palme, dor, short}` (after `best`/`film` stoplist
drop) intersect to only the short-film entry.

### Apostrophe / punctuation variant

`Critics Week Grand Prize` (44) vs `Critics' Week Grand Prize`
(39) normalize identically to `critics week grand prize`. They
collapse to a **single** `award_name_entry` row. `movie_awards`
rows with either raw form both point at the same
`award_name_entry_id`, so any query for either phrasing
retrieves the full union transparently.

### Cross-festival homonym (Grand Jury Prize)

Cannes has `Jury Prize` (66); Sundance has `Grand Jury Prize`
(1,276). These normalize to different strings and live in
different entry rows, so there's no homonym collision in the
entry table at all. If two ceremonies ever did share a byte-
identical normalized string, the `movie_awards.ceremony_id`
filter in Step 4 (applied whenever the ceremony channel has
resolved a ceremony) keeps the wrong-festival rows from
surfacing.

### Long-tail festival prize

`Waldo Salt Screenwriting Award` (37, Sundance only). LLM emits
`"waldo salt screenwriting award"`. Tokens `{waldo, salt,
screenwriting}` (after `award` stoplist drop) → single entry.
Works without any curation.

### LLM emits wrong form (BAFTA vs BAFTA Film Award)

User: "BAFTA winners". LLM emits any of `"bafta"`, `"bafta
award"`, `"bafta film award"`. After stoplist drops
`award`/`film`, all collapse to `{bafta}` → entries covering
`BAFTA Film Award`, `BAFTA TV Award`, `BAFTA Children's Award`.
Ceremony scoping plus the rest of the query (category, year)
narrows.

### IMDB surface-form inconsistency

If IMDB rescrape introduces a new variant, each becomes its own
`award_name_entry` row and enters the token index on the next
Stage A+B pass. No manual reconciliation.

### Historically distinct prizes (Grand Prize of the Festival vs
### Palme d'Or)

Cannes's pre-1955 `Grand Prize of the Festival` is a *different*
prize from the post-1955 `Palme d'Or`, not a rename — IMDB treats
them as distinct and so do we. Token sets are disjoint after
stoplist (`{grand, festival}` vs `{palme, dor}`), so the two
don't bridge. A user naming one doesn't get the other. Correct
behavior.

### Deep-cult historical prize

`Mussolini Cup` (130 tags, early Venice prize). LLM emits the
name; tokens `{mussolini, cup}` hit the entry directly. Works
without curation.

## Worked Examples

### Example 1 — "Oscar winners"

```
thinking: "Single flagship ceremony prize; Oscar is the base form."
award_names: ["oscar"]
→ tokens {oscar}
→ entries: {Oscar, Oscars Cheer Moment, Oscars Fan Favorite}
→ Step 4: entry_id IN (...) AND ceremony_id = OSCAR
→ all Oscar-tagged movie_awards rows.
```

### Example 2 — "Palme d'Or winners"

```
thinking: "Cannes flagship prize; base form is Palme d'Or."
award_names: ["palme d'or"]
→ normalize "palme dor" → tokens {palme, dor}
→ entries: {Palme d'Or, Palme d'Or - Best Short Film,
            Palme d'Or Spéciale}
→ Step 4: entry_id IN (...) AND ceremony_id = CANNES
→ 1262 + 44 + 1 = 1307 award rows.
```

### Example 3 — "Sundance Grand Jury Prize"

```
thinking: "Sundance's top prize; ceremony resolved separately."
award_names: ["grand jury prize"]
→ tokens {grand, jury} (prize in stoplist)
→ entries: {Grand Jury Prize, Short Film Grand Jury Prize,
            World Cinema Grand Jury Prize}
→ Step 4: entry_id IN (...) AND ceremony_id = SUNDANCE
  — the Sundance row-level filter excludes any Cannes
  "Jury Prize" rows that might otherwise leak via homonym.
```

### Example 4 — "Critics Week Grand Prize"

```
thinking: "Cannes Critics' Week top prize."
award_names: ["critics week grand prize"]
→ tokens {critics, week, grand}
→ entries: {critics week grand prize}  -- single merged entry
→ Step 4: entry_id IN (...) AND ceremony_id = CANNES
  — movie_awards rows with either raw form share the
  same entry_id, so the apostrophe split is handled at
  ingest, not query time.
```

### Example 5 — User asked for a specific sub-variant

```
User: "Best Short Film at Cannes"
thinking: "User named the short-film sub-variant explicitly."
award_names: ["palme d'or best short film"]
→ tokens {palme, dor, short} (best/film in stoplist)
→ entries: {Palme d'Or - Best Short Film}
→ Step 4: entry_id IN (...) AND ceremony_id = CANNES
Specificity preserved — main Palme d'Or correctly excluded.
```

## Open Decisions

1. **DF ceiling vs stopword droplist — resolved.** Post-backfill
   DF distribution (585 entries, 621 distinct tokens) is tri-modal
   and overlapping — no numeric threshold separates boilerplate
   from discriminative signal. Using a closed, hand-curated
   droplist applied at query time only. See the Stopword Droplist
   section above for the finalized list and the full rationale.
2. **Stoplist evolution — resolved.** Initial list finalized from
   the top-25 DF scan (see Stopword Droplist section). Revisit
   cadence: inspect the top 25 rows of
   `lex.award_name_token_doc_frequency` after every ingest sweep.
   Any list change is query-side-only — no re-ingest needed,
   since ingest writes every token.
3. **Ceremony filter precedence — resolved.** Ceremony scoping is
   applied only as a row-level filter on
   `movie_awards.ceremony_id` in Step 4. The entry table is
   ceremony-agnostic. If the ceremony channel returns empty, the
   ceremony filter is simply omitted and token resolution runs
   unscoped — the same behavior the rest of the award endpoint
   already uses for its other optional filters.

---

# Franchise Resolution

## Context

`public.movie_franchise_metadata` stores `lineage` and
`shared_universe` as independent TEXT columns and
`recognized_subgroups` as `TEXT[]`. Stage-3 retrieval does strict
string equality on each — see
[fetch_franchise_movie_ids](db/postgres.py#L1897). The failure
mode is documented in the Related Endpoints analysis: two LLMs
run months apart on different movies produce subtly different
canonical strings for the same concept, and exact match misses.

The fix ports the freeform half of the studio resolver —
symmetric normalization + token inverted index — with four
franchise-specific adjustments:

1. **Lineage and shared_universe are one search space, not two.**
   The ingest-side LLM sometimes places a franchise name in
   `lineage` and sometimes in `shared_universe` depending on
   scope judgment. The retrieval path must not require the
   query-side LLM to predict which column the stored name landed
   in. The two columns are preserved on the row for debugging and
   non-search consumers, but the search-time representation
   unions their entry ids into a single array.
2. **No registry, no enum.** The lineage / universe space is
   unbounded and grows with every new franchise. Maintenance cost
   of a closed enum outweighs the benefit; the token index
   handles umbrella and specific queries uniformly.
3. **Subgroups match by token intersection.** Currently exact
   match; promoted to the same token-intersection treatment as
   the combined lineage/universe field, which cleans up
   `phase 1` ↔ `phase one` / `snyderverse` ↔ `snyder-verse`
   style drift.
4. **Stopword droplist (not DF ceiling).** Non-discriminative
   English stopwords (`the`, `of`, `and`, `a`, `in`, `to`, `on`,
   `my`, `i`, `for`, `at`, `by`, `with`) are dropped symmetrically
   at ingest and query time. **Domain scaffolding** tokens
   (`trilogy`, `collection`, `films`, `series`, `universe`,
   `cinematic`, etc.) are **kept**. See "Why Not a DF Ceiling"
   below for the decision rationale.

## Design Principles

1. **Decouple lineage vs shared_universe at retrieval time.**
   Search matches against `franchise_name_entry_ids`, the union
   of the two columns' entry ids. A franchise name hits regardless
   of which column the ingest LLM chose.
2. **Symmetric normalization on both sides.** Reuse
   `normalize_string` + the shared ordinal and cardinal
   number-to-word rules defined on the studio side. Cardinal
   conversion (pure-numeric tokens 0–99) is what gets the
   `phase 1` ↔ `phase one` and `fast 2 furious` ↔ `fast two
   furious` cases across both ingest and query.
3. **Specificity via multi-token intersection.** Same rule as
   studios — multi-token names narrow; single-token names sweep
   the family.
4. **No registry, no alias table.** The ingest-side
   canonical-naming rules in
   [prompts/franchise.py](movie_ingestion/metadata_generation/prompts/franchise.py)
   already constrain emission. Token intersection forgives the
   residual orthographic drift.

## Data Model

```
franchise_entry
  id          PK
  normalized  TEXT UNIQUE   -- post-normalization canonical form;
                            -- the only key used at search time

franchise_token
  token                  TEXT
  franchise_entry_id     INT FK
  PRIMARY KEY (token, franchise_entry_id)

movie_franchise_metadata
  movie_id                      BIGINT PK
  lineage                       TEXT     -- kept for debug/analytics
  shared_universe               TEXT     -- kept for debug/analytics
  recognized_subgroups          TEXT[]   -- kept for debug/analytics
  franchise_name_entry_ids      INT[]    -- UNION of lineage +
                                         -- shared_universe entry ids
                                         -- GIN-indexed, USED AT SEARCH
  subgroup_entry_ids            INT[]    -- recognized_subgroups
                                         -- entry ids. GIN-indexed.
  -- lineage_position, is_spinoff, is_crossover, launched_subgroup,
  -- launched_franchise unchanged
```

**franchise_entry** — one row per distinct **normalized** string
seen across `lineage`, `shared_universe`, or any element of
`recognized_subgroups`. Normalization runs symmetrically at ingest
and query, so two columns whose raw strings differ only in casing
or punctuation collapse to a single id here. Lineage, universe,
and subgroup strings share this table: we never filter by where a
string came from at retrieval time, so separating them buys
nothing.

**franchise_token** — inverted index. GIN index on `token`.

**movie_franchise_metadata.franchise_name_entry_ids** — **union of
the lineage entry id and the shared_universe entry id** for the
movie. Two ids maximum (often one). This is the key
simplification — at search time, we match against this array and
don't care which column produced which id.

**movie_franchise_metadata.subgroup_entry_ids** — one id per
subgroup string on the movie.

The original `lineage`, `shared_universe`, `recognized_subgroups`
columns are preserved. They support debugging, movie-card
display, other query paths (`is_spinoff` / `shared_universe`
semantics), and re-ingestion if the index needs to be rebuilt.

## Ingestion Pipeline

### Stage A — Populate `franchise_entry`

Sweep `movie_franchise_metadata`. For each non-null `lineage`,
each non-null `shared_universe`, and each element of
`recognized_subgroups`, compute `normalized` via the shared
normalization rules and upsert a row in `franchise_entry` keyed
on `normalized`. Strings from different source columns that
collapse to the same normalized form resolve to the same row
(and therefore the same id) — this is the point. The stored
strings are already normalized at Stage-6 metadata generation
time; recomputing here guarantees symmetry with query time
regardless of drift between the Stage-6 prompt and the resolver.

### Stage B — Compute Token Document Frequencies (diagnostic)

Materialize `lex.franchise_token_doc_frequency` — one row per
distinct token with `COUNT(*)` over `franchise_token`. DF is
measured per `franchise_entry`, not per movie.

This view is **diagnostic**, not a filter. It exists to surface
new stopword candidates as the corpus grows, so the droplist can
be curated by hand from real data instead of guessed up front.
See "Why Not a DF Ceiling" below.

### Stage C — Stopword Droplist

`tokenize_franchise_string` drops this closed list symmetrically
on both sides (ingest and query):

```
the, of, and, a, in, to, on, my, i, for, at, by, with
```

All other tokens — including domain scaffolding like `trilogy`,
`collection`, `films`, `series`, `universe`, `cinematic`,
`chronicles`, `anthology`, `franchise` — are kept. The scaffolding
words overlap in DF range with legitimately discriminative words
(see rationale below), so any filter that would remove them also
removes signal we need.

Revisit cadence: after every ingest sweep, read the top 25 rows
of `lex.franchise_token_doc_frequency`. If a new English
stopword surfaces above the existing list, add it; if a
scaffolding word crosses a threshold where it's clearly dominating
retrieval, consider whether to drop it as a one-off rather than
by rule. Re-run Stages A and D across all movies after any
droplist change (cheap, no LLM cost).

#### Why Not a DF Ceiling

The initial design used a numeric DF ceiling. The post-backfill
DF distribution (5795 franchise entries, 5999 distinct tokens)
showed why that fails:

```
the        1005   ← stopword
of          294   ← stopword
trilogy     212   ← scaffolding
and         206   ← stopword
collection  107   ← scaffolding
films       107   ← scaffolding
a           106   ← stopword
in           76   ← stopword
man          58   ← DISCRIMINATIVE (Spider-Man, Iron Man, Batman, Ant-Man)
one          50   ← number word (ambiguous)
three        48   ← number word
dead         38   ← DISCRIMINATIVE (Evil Dead, Dead Poets)
love         36   ← DISCRIMINATIVE
christmas    32   ← DISCRIMINATIVE
black        30   ← DISCRIMINATIVE (Black Panther, Men in Black)
```

The distribution is **tri-modal**, and the bands overlap:

- Stopwords (frequency ≥ 76) — pure noise.
- Scaffolding (`trilogy` 212, `collection` 107, `films` 107,
  `series` 35) — meaningless alone but they're fine to keep as
  tokens, because a multi-token name always pairs them with the
  real franchise token in the intersection.
- Discriminative words in the 25–60 range — `man` (58) is load-
  bearing for the entire superhero corpus.

No single numeric threshold separates these. Cut at 100 and we
keep `trilogy`/`collection` (fine, but arbitrary); cut at 50 and
we kill `man`, breaking every Spider-Man / Iron Man / Ant-Man
query. The bad tokens are **semantically bad, not statistically
bad** — they need to be named, not counted.

With only ~6k tokens total, a closed, hand-curated stopword list
is cheap. The materialized DF view stays so the list can be
revisited from real data.

### Stage D — Build `franchise_token`

For every row in `franchise_entry`, tokenize again (same rules
as Stage B), drop any token that matches the stopword droplist
from Stage C, and insert one row into `franchise_token` per
surviving `(token, franchise_entry_id)` pair.

Create a GIN index on `franchise_token.token`.

### Stage E — Stamp movies

For each movie:

1. Resolve `lineage` (if non-null) and `shared_universe` (if
   non-null) against `franchise_entry` via `normalized` →
   up to two ids → write union to `franchise_name_entry_ids`.
2. Resolve each element of `recognized_subgroups` → write to
   `subgroup_entry_ids`.

GIN indexes on both columns.

### Stage F — Re-run cadence

Re-run on new movie ingest. If the normalizer or DF ceiling
changes, re-run Stages B–E across all movies (cheap; no LLM
cost). Re-running does not require regenerating Stage-6
metadata.

## Query-Time Resolution

The stage-3 franchise translator emits:

```python
class FranchiseQuerySpec(BaseModel):
    thinking: str                         # scope reasoning
    franchise_names: list[str]            # up to 3; covers lineage
                                          # AND shared_universe
                                          # (single combined space)
    recognized_subgroups: list[str]       # up to 3 subgroup names
    # structural flags + lineage_position unchanged
```

Key rename: `lineage_or_universe_names` → `franchise_names`,
reflecting the combined search space.

**Prompt contract:**

- Each `franchise_names` entry is a surface form the ingest-side
  LLM plausibly produced. Use the same canonical-naming rules
  already codified in `prompts/franchise.py` (lowercase, digits
  as words, abbreviation expansion where the expanded form is
  common).
- Emit the **broadest form** for umbrella queries (`"marvel
  cinematic universe"`, `"the lord of the rings"`); emit the
  **specific form** for narrow queries (`"doctor strange"`,
  `"captain america"`). Same specificity-by-user-intent rule as
  studios.
- Up to 3 alternates only when there are genuinely different
  canonical forms in wide use (`"marvel cinematic universe"` AND
  `"marvel"`; `"the lord of the rings"` AND `"middle-earth"`).
  Do not pad with spelling variants — normalization handles those.
- `thinking` field comes first and must commit scope (umbrella
  universe vs specific lineage vs sub-phase) before field values.

### Step 1 — Normalize and tokenize each name

For each `franchise_names` and each `recognized_subgroups`:

1. Apply `normalize_string` + ordinal + cardinal number-to-word.
2. Tokenize on whitespace AND hyphens.
3. Drop tokens matching the Stage C stopword droplist (applied
   symmetrically here). If all tokens drop, name contributes
   nothing.

### Step 2 — Per-name intersection

For each token in a name, fetch posting list from
`franchise_token`. Intersect posting lists within the name →
`franchise_entry_id` set.

### Step 3 — Across-name union

Union per-name entry-id sets separately for:

- `franchise_names` → `franchise_entry_ids_A`
- `recognized_subgroups` → `franchise_entry_ids_B`

### Step 4 — Resolve to movies

```sql
SELECT movie_id FROM movie_franchise_metadata
WHERE franchise_name_entry_ids && :franchise_entry_ids_A
  AND (
    :franchise_entry_ids_B = '{}'
    OR subgroup_entry_ids && :franchise_entry_ids_B
  )
```

`&&` is the GIN array-overlap operator. When both franchise names
and subgroups are present the two constraints AND (user asked for
"MCU Phase One movies" → must belong to MCU AND to Phase One).

Structural flags (`is_spinoff`, `is_crossover`,
`launched_subgroup`, `launched_franchise`) and `lineage_position`
filter the result set via their existing columns, unchanged.

### Step 5 — No further fallbacks

If token resolution returns empty, the franchise channel
contributes nothing.

## Normalization Rules

Shared with studio and title:

1. Lowercase.
2. Diacritic-fold (NFKD → strip combining marks).
3. Strip punctuation.
4. Collapse whitespace.
5. Ordinal number-to-word (`phase 1st` → `phase first`) — shared
   rule. Always on, both sides.
6. **Cardinal number-to-word** for pure-numeric tokens with
   integer value in [0, 99] (`phase 1` → `phase one`, `fast 2
   furious` → `fast two furious`). Same shared rule as studio
   and awards — bounds, leading-zero handling, and the "numbers
   ≥ 100 stay as digits" carve-out all apply here identically.
   Always on, both sides. Applies across lineage,
   shared_universe, and subgroup name strings. The ingest prompt
   already mandates word form, but we do not rely on that — the
   normalizer is the source of truth, and running it symmetrically
   removes any dependence on prompt discipline.

**Tokenization: whitespace AND hyphens.** Same rule as studio and
title.

## Edge Cases

### Ingest put the brand in one column, query emits the other

Ingest wrote `shared_universe = "marvel cinematic universe"`,
`lineage = "captain america"` for Captain America: The First
Avenger. User asks "MCU movies" → LLM emits
`franchise_names = ["marvel cinematic universe"]`. Token lookup
resolves to the MCU entry id. The movie's
`franchise_name_entry_ids` contains that id (from the
`shared_universe` side of the union). Match. The fact that MCU
came from `shared_universe` rather than `lineage` is invisible to
the search path.

### Stopword drift (the lord of the rings)

Ingest wrote `"the lord of the rings"`, query LLM emits `"lord of
the rings"`. `the` and `of` are in the stopword droplist and are
dropped on both sides. Both strings reduce to `{lord, rings}` and
resolve to the same entry id.

### Digit-vs-word drift (phase 1 vs phase one)

Cardinal number-to-word normalization collapses both to `phase
one` on both sides. Same entry id.

### Hyphen-split drift (spider-man vs spider man)

Ingest stores `"spider-man"` (prompt mandates x-men/spider-man
stays hyphenated). Hyphen-split tokenizer → `{spider, man}`.
Query `"spider-man"` or `"spider man"` both tokenize to `{spider,
man}` → match. Query `"spiderman"` is one token `{spiderman}` and
does not bridge — prompt discipline on the query side is the
mitigation, matching what ingest does.

### Subgroup fragmentation (phase one vs infinity saga)

Both exist as distinct subgroup strings. Token intersection DOES
NOT bridge them — they are legitimately different sub-phases.
User asking "MCU phase one" gets phase-one movies; user asking
"infinity saga" gets infinity-saga movies. If the ingest LLM
inconsistently tagged the same movie with one vs the other,
that's an ingest-side data-quality issue, not a retrieval issue.

### Umbrella sweep (Marvel → Marvel Cinematic Universe)

User: "Marvel movies". LLM emits `franchise_names = ["marvel
cinematic universe", "marvel"]`. First tokens `{marvel, cinematic,
universe}` — per-name intersection resolves to the single MCU
entry (only entry containing all three). Second tokens `{marvel}`
— posting list sweeps every entry carrying `marvel` as a token
(`marvel cinematic universe`, `marvel comics`, `marvel knights`,
etc.). Across-name union → MCU id plus every Marvel-flavored
entry id. Matches MCU films plus other Marvel-stamped films. The
umbrella sweep works by emitting the bare-brand alternate, not by
filtering scaffolding out of the universe form.

### Long-tail lineage with no universe

User: "James Bond movies". Ingest wrote `lineage = "james bond"`,
`shared_universe = null`. LLM emits `["james bond"]`. Tokens
`{james, bond}` → Bond entry id. Movie's
`franchise_name_entry_ids` contains it from the lineage side of
the union.

### Acronym that must expand (MCU)

Ingest prompt canonicalizes `MCU` → `marvel cinematic universe`.
Query-side prompt must do the same. If the query LLM emits
`"mcu"` anyway, tokens `{mcu}` don't appear in any posting list.
Prompt guidance is the mitigation; no code-level acronym
expansion.

### LLM emits wrong specificity

User: "Marvel". LLM emits `["doctor strange"]` instead of
`["marvel cinematic universe"]`. The `thinking` field catches
most of these. When it doesn't, retrieval is too narrow; vector
and metadata channels carry the search. Browsing UX absorbs the
imprecision — same fallback philosophy as studios.

### Franchise-names OR semantics when user wanted AND

User: "Doctor Strange in the MCU". If the LLM emits both
`["doctor strange", "marvel cinematic universe"]`, the
across-name union produces OR semantics and retrieval is too
broad (all MCU movies, not just Doctor Strange). **Mitigation
lives in the prompt:** when the user names a specific lineage
inside a universe, the LLM emits only the narrower form since
every Doctor Strange film is already MCU. The `thinking` field
commits this before field values.

## Worked Examples

### Example 1 — "Marvel movies"

```
thinking: "Broad umbrella. Use the universe form plus the bare
           brand token."
franchise_names: ["marvel cinematic universe", "marvel"]
recognized_subgroups: []
→ name 1 tokens {marvel, cinematic, universe} → MCU entry id
→ name 2 tokens {marvel} → every entry carrying `marvel`
  ({marvel cinematic universe, marvel comics, marvel knights, ...})
→ across-name union → MCU id ∪ all Marvel-stamped entry ids
→ franchise_name_entry_ids && :ids
→ all MCU movies plus other Marvel-stamped films.
```

### Example 2 — "Doctor Strange in the MCU"

```
thinking: "Narrow lineage inside a universe. Every Doctor Strange
           film is MCU, so the narrow form alone suffices."
franchise_names: ["doctor strange"]
recognized_subgroups: []
→ tokens {doctor, strange}
→ single doctor strange entry id
→ Doctor Strange films only.
```

### Example 3 — "MCU Phase One"

```
thinking: "Umbrella PLUS specific sub-phase."
franchise_names: ["marvel cinematic universe"]
recognized_subgroups: ["phase one"]
franchise_entry_ids_A = [MCU id]
franchise_entry_ids_B = [phase one id]
SQL: franchise_name_entry_ids && [MCU]
     AND subgroup_entry_ids && [phase one]
→ Iron Man, Incredible Hulk, Iron Man 2, Thor, Captain America,
  Avengers.
```

### Example 4 — "Lord of the Rings"

```
thinking: "Long-running franchise, umbrella query."
franchise_names: ["the lord of the rings"]
recognized_subgroups: []
→ tokens {lord, rings} (the/of dropped by stopword droplist)
→ lord of the rings entry id
→ Jackson trilogy (stored as lineage=the lord of the rings).
```

### Example 5 — Jackson-specific narrowing

```
User: "Jackson's LOTR trilogy"
thinking: "Director-era subgroup named explicitly."
franchise_names: ["the lord of the rings"]
recognized_subgroups: ["jackson lotr trilogy"]
subgroup tokens {jackson, lotr, trilogy} (scaffolding kept —
                                  `trilogy` survives alongside
                                  the discriminative tokens)
→ jackson lotr trilogy entry id (intersection of all three
   posting lists resolves to the single subgroup entry)
→ Only Peter Jackson's LOTR films, not Bakshi animated, not the
  Hobbit trilogy.
```

### Example 6 — Phase-number word/digit bridging

```
User: "Phase 1 Marvel movies"
Cardinal normalization: "phase 1" → "phase one" on the query side.
franchise_names: ["marvel cinematic universe"]
recognized_subgroups: ["phase one"]
Matches ingest-side stored "phase one" directly.
```

### Example 7 — Ingest stored the brand in shared_universe only

```
User: "Conjuring Universe movies"
Ingest rows:
  The Conjuring → lineage="the conjuring",
                  shared_universe="conjuring universe"
  Annabelle     → lineage="annabelle",
                  shared_universe="conjuring universe"
franchise_names: ["conjuring universe"]
→ tokens {conjuring, universe} (scaffolding kept)
→ per-name intersection → single `conjuring universe` entry id
→ franchise_name_entry_ids && [conjuring_universe_id]
  - The Conjuring ✓ (conjuring universe id present via shared_universe)
  - Annabelle ✓ (conjuring universe id present via shared_universe)
  - Nun, Curse of La Llorona, etc. ✓
The combined-column representation is what makes this work —
Annabelle's `lineage` is "annabelle" but its
`franchise_name_entry_ids` includes the `conjuring universe` id
from the `shared_universe` side. The umbrella sweep across the
whole Conjuring family (including the standalone `the conjuring`
lineage entry) depends on the ingest-side union, not on stripping
`universe` at query time.
```

## Open Decisions

1. **Stopword droplist vs DF ceiling (resolved).** Franchise-side
   filtering uses a closed stopword list
   (`the, of, and, a, in, to, on, my, i, for, at, by, with`),
   **not** a numeric DF ceiling. Rationale: the post-backfill DF
   distribution is tri-modal (stopwords, scaffolding,
   discriminative words), and scaffolding (`trilogy` 212,
   `collection` 107, `films` 107) overlaps in DF range with load-
   bearing words like `man` (58, Spider-Man/Iron Man/Batman) and
   `dead` (38). No single threshold separates the three bands, so
   a hand-curated droplist is used instead. The materialized DF
   view is retained as a diagnostic for revisiting the droplist as
   new data comes in. See Stage C, "Why Not a DF Ceiling" for the
   full decision.
2. **`franchise_names` OR vs AND semantics.** Across-name union
   is OR. For the specific "narrow lineage inside umbrella"
   case, prompt discipline produces the right narrow form
   alone. If evaluation shows this pattern misfires at scale,
   consider a narrower-of-two selection rule in the resolver.
3. **Rebuild-on-prompt-change.** When canonical-naming rules in
   `prompts/franchise.py` change, stored ingest-side canonical
   forms drift from new LLM emissions. Stages A–E rebuild the
   index without regenerating Stage-6 metadata. Confirm this
   workflow is cheap enough to re-run per prompt iteration.

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
applied at ingest when writing rows to
`lex.inv_production_brand_postings`. Star Wars 1977 is not a Disney
movie; Force Awakens 2015 is.

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
