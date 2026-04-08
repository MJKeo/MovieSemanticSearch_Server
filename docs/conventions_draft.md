# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /solidify-draft-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.

## Labeled embedding text for short categorical fields
**Observed:** User approved adding semantic labels ("genres: ", "conflict: ", "themes: ") to short categorical fields in embedding text while leaving prose fields unlabeled. Backed by research (Anthropic contextual retrieval, Google Gemini docs). Applied to PlotAnalysisOutput.embedding_text() and ReceptionOutput.embedding_text().
**Proposed convention:** When building text for vector embedding, prefix short/ambiguous categorical fields with a semantic label ("field_name: value1, value2"). Leave prose fields (summaries, overviews) unlabeled — they are self-contextualizing. This helps the embedding model disambiguate the role of short values.
**Sessions observed:** 2

## Per-term normalization in embedding text, not whole-string
**Observed:** User directed normalizing each individual term before joining with ", " rather than running normalize_string() on the entire assembled embedding text. Applied to PlotAnalysisOutput.embedding_text() and ReceptionOutput.embedding_text() label fields.
**Proposed convention:** In embedding_text() methods, normalize each label/term individually before joining. Do not run normalize_string() on the final assembled string — this preserves label prefixes and structural formatting (colons, commas) while still normalizing the actual content.
**Sessions observed:** 2

## Metadata classes own their embedding representation
**Observed:** User explicitly directed moving embedding formatting logic from vector_text.py into embedding_text() methods. Applied to PlotAnalysisOutput ("move that to within the embedding_text method... that's a much cleaner approach") and ReceptionOutput. The vector_text functions became thin wrappers.
**Proposed convention:** Each *Output metadata class should own its embedding text assembly in its embedding_text() method. Vector text functions in vector_text.py should be thin wrappers that may enrich (e.g., merge genres, add tier) but delegate core formatting to the metadata class.
**Sessions observed:** 2

## No numeric scores in vector embedding text
**Observed:** User explicitly said "Do not add numbers into the vector" when Claude proposed including IMDB rating and Metacritic score alongside the reception tier label. Numeric scores are metadata filter territory, not embedding territory.
**Proposed convention:** Never include raw numeric scores (ratings, vote counts, years, budgets) in vector embedding text. Use qualitative labels derived from scores (e.g., "universally acclaimed" from a score >= 81) instead. Numeric filtering belongs in metadata scoring, not vector similarity.
**Sessions observed:** 1

## json_each() for SQLite batch WHERE IN clauses
**Observed:** User pointed to Movie._BATCH_QUERY using `WHERE tmdb_id IN (SELECT value FROM json_each(?))` and asked why _mark_ingested didn't use the same pattern instead of executemany. Applied to _mark_ingested — now matches.
**Proposed convention:** For SQLite queries that filter on a set of IDs, use `WHERE column IN (SELECT value FROM json_each(?))` with `json.dumps(list_of_ids)` as the single bound parameter. This avoids placeholder string building, sidesteps SQLITE_MAX_VARIABLE_NUMBER limits, and keeps the query text constant for plan caching. Prefer over both executemany loops and `IN (?, ?, ...)` expansion.
**Sessions observed:** 1

## No re-export shims when moving modules
**Observed:** User explicitly rejected keeping old files as re-export shims after moving schemas to a new package. Said "No hacking by re-exporting moved files in their original file. Update at the source of each import directly."
**Proposed convention:** When moving a module to a new location, update all import sites directly. Never leave the old file as a re-export shim — delete it and fix every consumer.
**Sessions observed:** 1

## Per-call async clients for connection-pooled HTTP SDKs
**Observed:** AsyncOpenAI singleton caused silent hangs due to httpx connection pool exhaustion after N calls. Fix was `async with AsyncOpenAI(...) as client:` per call instead of reusing a module-level instance. The sync client was unaffected.
**Proposed convention:** For async SDK clients that manage HTTP connection pools internally (OpenAI, httpx-based), prefer per-call instantiation with `async with` over module-level singletons — especially when the client is called repeatedly in batch loops. Sync clients can remain singletons. If a singleton is used, configure an explicit timeout so pool exhaustion surfaces as an error rather than a silent hang.
**Sessions observed:** 1

## GIN arrays for enum-ID filtering, inverse posting tables for text-matched entities
**Observed:** During V2 data architecture design, the user asked whether movie_card array columns (genres, languages, etc.) need inverse lookup tables or just GIN-indexed array overlap. The clear principle emerged: GIN `&&` with `gin__int_ops` is correct for enum-like IDs where you filter by set membership (genres, languages, providers, keywords, countries, source material types). Inverse posting tables are correct for text-matched entities that need fuzzy/trigram lookup, efficient reverse lookup (term→movies), or per-pair metadata like billing_position (people, characters, studios, titles, franchises). The codebase already follows this split but the principle wasn't articulated.
**Proposed convention:** Use GIN-indexed INT[] columns on movie_card for enum-based set-membership filtering (`&&` overlap). Use inverse posting tables in lex schema for entities that require fuzzy text matching at the entry point, reverse lookup (given entity → find movies), or per-pair metadata. Never create inverse posting tables for enum-only fields; never use GIN arrays for text-matchable entities.
**Sessions observed:** 1

## Canonical naming convention for LLM-generated matchable strings
**Observed:** User corrected a proposed alias/abbreviation table approach for franchise names. Instead of generating aliases, both the ingestion LLM and search extraction LLM should be instructed to use the most common, fully expanded form — no abbreviations. Same pattern already used by the lexical entity extractor for person names. This ensures both sides converge on the same canonical string without needing alias infrastructure.
**Proposed convention:** When two LLMs must produce matching strings (one generating data, one extracting from queries), instruct both to output the most common, fully expanded form with no abbreviations. This eliminates the need for alias tables or enum vocabularies. Apply to any entity type stored in the lexical dictionary (franchises, person names, etc.).
**Sessions observed:** 1

## Search prompts must match actual embedded content, not assumed content
**Observed:** During vector search prompt realignment, found that search prompts (subquery generation + weight assessment) described content that didn't match what's actually embedded — stale field names, content from adjacent vector spaces, fabricated data sources (e.g., cast/crew names listed in production vector but not actually embedded). The realignment process required reading vector_text.py and embedding_text() methods as the source of truth, then rewriting prompts from scratch.
**Proposed convention:** Every content claim in a vector search prompt (subquery or weight) must trace to a specific field in the vector_text.py generation function or the metadata class's embedding_text() method. When metadata generation schemas change, the corresponding search prompts must be updated in the same changeset or explicitly tracked as a follow-up. vector_text.py is the single source of truth for what's embedded.
**Sessions observed:** 1
