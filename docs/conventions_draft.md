# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /solidify-draft-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.

## Concrete typed LLM output when format parity is required
**Observed:** When designing the step 3 semantic endpoint, weighed freeform `query_text` strings vs. concrete per-space Pydantic objects matching ingestion-side embedding shape. Chose concrete objects because freeform strings make the LLM the only enforcement of format parity — a silent drift risk. Concrete objects wrapped in discriminated unions (`Literal[space]` on each wrapper) bind the format at the schema layer; cross-side format drift becomes structurally impossible.
**Proposed convention:** When LLM output must match a format produced elsewhere in the system (e.g., ingestion-side embedding text that query-side vectors need to align with), emit concrete typed objects, not freeform strings. The Pydantic schema should mirror the target format's shape closely enough that a shared or equivalent `embedding_text()` / formatter method on both sides produces byte-identical output.
**Sessions observed:** 1

## Duplicate embedding/format logic deliberately over factoring into shared helpers
**Observed:** When designing query-side `*Body` classes that mirror ingestion-side `*Output.embedding_text()`, considered factoring the shared formatter into a helper. Chose to duplicate instead, on the reasoning that duplication surfaces divergence in code review whereas a shared helper would silently change both sides' behavior on refactor.
**Proposed convention:** When two sides of the system must produce identical formatted output but evolve independently (different data sources, different validation needs), duplicate the formatting logic rather than factoring into a shared helper. Document the duplication intent in-file. This converts silent drift into a code-review-visible diff.
**Sessions observed:** 1

## Query-side schemas shouldn't mechanically replicate ingestion-side data-hygiene guards
**Observed:** `ProductionBody` initially replicated ingestion's `filming_locations[:3]` truncation. User corrected: the cap is ingestion-side defense against scraped noise; on the query side the LLM emits only intentional locations, so silently truncating a 4th entry discards real signal. The underlying principle: a guard that's correct on one side can be wrong on the other when the data sources and failure modes differ.
**Proposed convention:** When mirroring schemas across pipeline stages (ingestion vs. query, raw vs. synthesized), evaluate each guard/transformation's reason-for-being before carrying it across. Guards against upstream data noise belong only on the ingestion side. Guards enforcing invariants the downstream consumer depends on belong on both sides. Don't copy the guard just because the shape is the same.
**Sessions observed:** 1

## Edge-case guards belong at the narrowest caller, not in shared helpers
**Observed:** When fixing a lone-hyphen token leaking into `studio_token`, scoped the guard to `tokenize_company_string` (studio-only wrapper) rather than modifying the shared `tokenize_title_phrase`. Motivation: title tokenization has been stable and the same "bare `-` → drop" transform hasn't been validated for titles; changing the shared helper would silently alter every existing lex.lexical_dictionary consumer. User accepted the narrower scope without pushback and the docstring explicitly called it out ("without altering the shared title tokenizer's behavior"). Same instinct surfaced in the module docstring ("The shared `normalize_string` in helpers.py is intentionally left alone — rewriting it globally would invalidate every existing lex.lexical_dictionary row").
**Proposed convention:** When a defensive guard or transform is correct for one caller but has not been verified to be correct for every caller of a shared helper, apply the guard at the narrow caller — a wrapper, post-filter, or specialized helper — not by editing the shared helper in place. Shared helpers' invariants are load-bearing for every downstream index and cache; change them only when the new behavior is deliberately required across every caller, and when the cost of a follow-up rebuild of all derived data is understood.
**Sessions observed:** 1

## Preserve source columns alongside derived search arrays
**Observed:** Both the studio resolver (keeps raw `production_company.canonical_string` strings alongside the derived `movie.brand_ids INT[]` stamp) and the new franchise design (keeps raw `lineage` / `shared_universe` / `recognized_subgroups` TEXT columns alongside the derived `franchise_name_entry_ids INT[]` / `subgroup_entry_ids INT[]` arrays) follow the same pattern: search-time access uses the denormalized ID arrays, but the original LLM-authored TEXT columns remain on the row. User explicitly directed this for franchise: "We can keep the older columns for debugging / easily seeing what is what for a given movie, but for searching it should be one list."
**Proposed convention:** When introducing a denormalized / projected representation for search (GIN-indexed entry-id arrays, stamped brand arrays, token-index entries), do not drop the original source columns. Keep the canonical human-readable TEXT form on the row. Reasons: (1) debugging — "what did the LLM actually write?" must be answerable without re-running generation; (2) rebuildability — stoplist/normalizer changes let you regenerate the index from the source columns without LLM re-generation; (3) non-search consumers (movie cards, analytics, evals) still read the human-readable form. The projected columns are derivable; the source is not.
**Sessions observed:** 1
