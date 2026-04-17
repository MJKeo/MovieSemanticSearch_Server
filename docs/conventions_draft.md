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
