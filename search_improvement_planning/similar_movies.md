# Similar Movies Search

A separate flow for unqualified "like X" queries (e.g. "like Inception"). Does not go through the standard search pipeline — has its own candidate generation and ranking.

## Goal

Given a single anchor movie, return movies that someone would genuinely consider "similar" — capturing both substance (themes, plot) and experience (tone, scale, prestige tier).

## Similarity Axes

Three independent candidate sources, each generates its own candidates and scores. A weaving/rescoring step combines them into the final order (strategy TBD).

### 1. Movie Shape (strong)

Vector similarity across all 8 spaces, plus quality/reception bucket matching.

**Vector spaces:** all 8, with extra weight on plot_analysis, viewer_experience, and dense_anchor (these carry most of the "what's it about and how does it feel" load).

**Quality / reception bucketing:** the anchor movie is classified into one of three buckets, and we match candidates to its bucket when applicable.

- **Bucket 1 — Poorly rated but still loved (cult).** Indicated by opposing reception and popularity stats (low reception score with disproportionate popularity / engagement). Match candidates from the same bucket.
  - Note: this signal also picks up mainstream guilty pleasures (e.g. Transformers) that aren't true cult films. We don't have critic-vs-audience divergence data to disambiguate, so vector matching has to carry that distinction.
- **Bucket 2 — Critically acclaimed.** High reception with popularity as a secondary factor (matters somewhat — separates art-house prestige from mainstream prestige, but reception dominates). Match candidates from the same bucket.
- **Bucket 3 — Middle of the road.** Doesn't match either extreme. Apply a generic quality / notability boost rather than bucket matching, since there's no strong taste signal to preserve.

Bucketing avoids the failure mode where pure vector similarity surfaces tonally-wrong matches (e.g. low-budget indie versions of prestige blockbusters, or prestige cinema in response to "like Sharknado").

### 2. Director (strong-moderate)

Lexical match on director. Single biggest non-vector signal — directors have signatures that vector spaces only partially capture.

Should not depend heavily on movie shape (a Nolan film is still a relevant "like Inception" result even if its plot diverges), but director and shape scores **are not mutually exclusive** — a movie that matches on both should rank higher than one that matches on either alone.

### 3. Franchise / Universe / Lineage (moderate)

Direct franchise / universe match (sequels, prequels, spinoffs, remakes). Worth weaving in but **should not be the top result unless it also matches the other axes well** — someone asking for "like Inception" probably doesn't want only Tenet and Interstellar; they want a broader similarity surface.

## Explicitly Out of Scope

These were considered and rejected for this flow:

- **Genre cluster** — already implicit in vectors; explicit weighting adds little.
- **Budget / production scale** — partially captured by the production vector and quality bucketing.
- **Lead actor overlap** — too weak (DiCaprio in Titanic is not "like Inception").
- **Same composer / DP** — real signal but too sparse to be worth the complexity.
- **Era / decade, country / language** — captured by vectors.

## Open Questions

- **Weaving / rescoring strategy.** How do we combine candidates from the three sources into a final ordered list? Options include score normalization + weighted sum, round-robin interleaving with tier constraints, or a learned reranker. TODO.
- **Quality bucket thresholds.** What concrete reception/popularity cutoffs define each bucket? Needs distribution analysis.
- **Cult signal robustness.** Without critic-vs-audience data, how reliably can reception-vs-popularity distinguish true cult films from mainstream guilty pleasures? Vector matching is expected to carry the slack, but worth validating.
- **Multi-anchor similarity ("like Inception *and* Memento").** Current design assumes a single anchor movie. Naive vector averaging is geometrically biased (pulls toward dense clusters, can fall off the embedding manifold, drowns out distinct flavors of diverse inputs) and not a principled "semantic average." Better approach is likely per-anchor search with score aggregation across candidates (mean / min / reciprocal rank fusion) rather than a single averaged-vector query. Director and franchise axes also need a multi-anchor combination story (intersection vs union). TODO.
