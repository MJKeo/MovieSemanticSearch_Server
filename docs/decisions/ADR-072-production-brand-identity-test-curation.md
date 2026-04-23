# [072] — Production brand registry: identity-test curation over corporate ownership

## Status
Active

## Context
The brand registry in `schemas/production_brands.py` included every
corporate subsidiary a parent brand had ever owned. This caused "Disney
movies" to return films like "No Country for Old Men" (a Miramax film
distributed under a Disney subsidiary). The original spec used a
"catalog recall over label purity" principle that conflated corporate
ownership with brand identity.

## Decision
Replace the curation principle with an identity test: **a label belongs
in a brand's roster only if a casual viewer typing `<brand> movies`
would expect its films.**

Practical rules applied across all 31 brands:
- KEEP labels the parent actively brand-promotes (e.g. Pixar under DISNEY
  as "Disney/Pixar", Marvel Studios under DISNEY, Lucasfilm under DISNEY).
- DROP autonomous-identity acquisitions (Miramax, Searchlight, Touchstone,
  Hollywood Pictures, Blue Sky under DISNEY; New Line, HBO under WB;
  Focus, DreamWorks Animation, Working Title, Nickelodeon, MTV under their
  respective parents; Sony Pictures Classics under SONY).
- DROP home-entertainment, foreign-region, and distribution-only credits.

Dropped labels remain findable via their own standalone brand entries —
"Miramax movies" still works; the change only stops those films from
leaking into their former parent's roster.

Multi-brand tagging is preserved where legitimately co-branded: Pixar
films still tag DISNEY + PIXAR; post-2022 MGM films still tag MGM +
AMAZON_MGM.

## Alternatives Considered
- **Retain corporate-ownership principle**: Technically complete but
  produces results users find wrong. "Disney movies" including Miramax
  horror films violates the user's mental model.
- **User-preference signal from query analysis**: "Disney movies" could
  be interpreted contextually. But the brand registry curation is an
  offline decision; query-time adjustment adds pipeline complexity and
  latency without fixing the root cause.

## Consequences
- The 31 brand rosters are now calibrated to user expectations rather
  than corporate org charts.
- `unit_tests/production_brand_spec_dates.py`, `test_brand_resolver.py`,
  and `test_production_brands.py` reference old rosters and need to be
  regenerated against the new membership lists.
- `schemas/production_brand_surface_forms.py` picks up changes
  automatically at import time (reads dynamically from the enum).
- `movie_ingestion/final_ingestion/rebuild_production_brand_postings.py`
  must be run after this change to refresh `lex.inv_production_brand_postings`
  without a full re-ingestion.

## References
- search_improvement_planning/production_company_tiers.md — original tier
  structure (preserved: 24 + 7 brands, slug/ID assignments, year-window
  mechanics)
- ADR-068-award-data-postgres-storage.md
