# [078] — Category taxonomy relocated to schemas/trait_category.py with richer per-member attributes

## Status
Active

## Context
`CategoryName` previously lived in `schemas/enums.py` alongside all
other search-side enums. As the taxonomy grew from 32 to 43+ active
categories and the category-handler scaffolding was built, two problems
emerged:

1. **Prompt-quality coupling.** The per-category descriptions in the
   enum needed to carry more than just a description string — effective
   handler dispatch requires boundary statements, edge cases, good/bad
   examples, endpoint routing, and bucket shape. Packing all of this
   into `enums.py` alongside unrelated generation enums would make both
   files harder to read and harder to update.

2. **Category-specific prompt assembly.** The step-2 system prompt and
   the handler prompt builder both construct category sections
   programmatically from the enum. Having the taxonomy in its own module
   with a clean import path makes the dependency explicit and the module
   boundary clear.

## Decision
Move `CategoryName` to a dedicated `schemas/trait_category.py` and
extend each member to carry seven attributes:

- `description` — what the category covers (LLM-facing).
- `boundary` — what the category does NOT own, with explicit redirects.
- `edge_cases` — concrete misroute traps with disambiguators.
- `good_examples` — surface forms that clearly belong.
- `bad_examples` — surface forms that look like they belong but route elsewhere.
- `endpoints` — ordered `EndpointRoute` tuple (priority order).
- `bucket` — `HandlerBucket` enum (SINGLE/MUTEX/TIERED/COMBO).

The `__new__` constructor takes all seven; the tuple-constructor pattern
is consistent with `NarrativeStructureTag`, `LineagePosition`, and
`AwardCeremony`.

The 45-slot numbering is preserved with gaps at 43 and 45 (removed
parametric-expansion categories) so cross-references in prior planning
docs remain valid.

## Alternatives Considered
- **Keep in schemas/enums.py, add more attributes**: Simpler import
  structure, but the file becomes unwieldy and the taxonomy's
  prompt-assembly role is obscured.
- **Separate registry module alongside the enum**: Registry-style dict
  mapping `CategoryName → handler metadata`. Rejected because it creates
  drift risk — two sources of truth for the same per-category facts.
  Baking attributes onto the enum makes them atomic: change the member,
  change all its attributes together.
- **Runtime-built attributes via a decorator or `__init_subclass__`**:
  More dynamic but loses the IDE-visible, diff-friendly per-member
  definition format.

## Consequences
- Any `from schemas.enums import CategoryName` will fail; import path
  is now `from schemas.trait_category import CategoryName`.
- Nine import sites were updated in the initial migration; any new
  file importing `CategoryName` must use the new path.
- The planning doc (`search_improvement_planning/query_categories.md`)
  and the enum are the dual sources of truth; the enum members should
  stay aligned with the doc. The doc is the design surface; the enum
  is the code surface.
- `HandlerBucket` and `EndpointRoute` remain in `schemas/enums.py`
  because they are used by modules other than the trait taxonomy.

## References
- schemas/trait_category.py
- schemas/enums.py (HandlerBucket, EndpointRoute)
- search_improvement_planning/query_categories.md
- docs/modules/schemas.md
