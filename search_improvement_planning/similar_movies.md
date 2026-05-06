# Similar Movies Search

A separate flow for unqualified "like X" queries (e.g. "like Inception"). Does not go through the standard search pipeline — has its own candidate generation and ranking.

## Goal

Given a single anchor movie, return movies that someone would genuinely consider "similar" — capturing both substance (themes, plot) and experience (tone, scale, prestige tier).

## Similarity Axes

Four independent candidate sources, each generates its own candidates and scores. A weaving/rescoring step combines them into the final order (strategy TBD).

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

### 4. Studio / Source Lineage (moderate auxiliary)

Production-brand / studio-company overlap and source-of-inspiration overlap are useful as **latent lineage signals**: they capture shared creative machinery, audience calibration, source tradition, or house style that semantic vectors may only partially infer from the movie text.

This signal is **not a replacement for semantic shape**. The vector spaces already capture much of the observable output of studio/source influence — tone, premise, pacing, themes, genre, production style. Studio/source lineage should therefore act as a tie-breaker, agreement boost, and limited recall-repair path, not as a dominant retrieval axis.

**Studio / production company:** Use only identity-bearing brands or specific company/studio labels whose name meaningfully predicts something about the movie. Broad corporate umbrellas are weak for similarity even when useful for explicit studio search.

- Stronger examples from the current brand registry: Pixar, Walt Disney Animation, Studio Ghibli, DreamWorks Animation, Illumination, Sony Pictures Animation, Marvel Studios, Lucasfilm, DC Studios, A24, Neon, Blumhouse, Searchlight, Focus Features, Miramax, Touchstone, New Line Cinema.
- Moderate / context-dependent examples: Lionsgate, MGM, United Artists, TriStar, Columbia, 20th Century, Apple Studios, Amazon MGM, Netflix.
- Weak examples for similarity: Disney umbrella, Warner Bros., Universal, Paramount, Sony. These are too broad; require strong vector agreement before they contribute, or ignore them for similarity scoring.

Studio identity can be **era-dependent**. For labels whose house style changes over time, apply an era-proximity multiplier rather than treating all same-brand matches equally. Examples: Walt Disney Animation classic / Renaissance / Revival / modern eras; Pixar early-golden vs sequel-heavy/recent eras; Marvel Studios pre-Endgame vs post-Endgame; Miramax's 1990s prestige identity; New Line's genre/indie vs commercial blockbuster phases.

**Source of inspiration:** Use generated source-of-inspiration metadata as a lineage signal when movies share adaptation tradition or source type. This is intentionally not exact-author matching because exact source author is not currently available in the DB. Source lineage should help surface movies that share myth, fairy tale, comic-book, folklore, historical-event, toy/game, remake, or other inspiration patterns even when their literal plot text differs.

## Explicitly Out of Scope

These were considered and rejected for this flow:

- **Genre cluster** — already implicit in vectors; explicit weighting adds little.
- **Budget / production scale** — partially captured by the production vector and quality bucketing.
- **Lead actor overlap** — too weak (DiCaprio in Titanic is not "like Inception").
- **Same composer / DP** — real signal but too sparse to be worth the complexity.
- **Era / decade, country / language** — captured by vectors.
- **Exact source author matching** — not currently available in the DB; use source-of-inspiration categories/patterns instead.

## Open Questions

- **Weaving / rescoring strategy.** How do we combine candidates from the four sources into a final ordered list? Options include score normalization + weighted sum, round-robin interleaving with tier constraints, or a learned reranker. TODO.
- **Quality bucket thresholds.** What concrete reception/popularity cutoffs define each bucket? Needs distribution analysis.
- **Cult signal robustness.** Without critic-vs-audience data, how reliably can reception-vs-popularity distinguish true cult films from mainstream guilty pleasures? Vector matching is expected to carry the slack, but worth validating.
- **Studio/source weighting.** How much should latent lineage boost candidates beyond semantic-vector agreement? Initial stance: moderate auxiliary signal only; prevent broad studios or weak source overlap from outranking better semantic matches.
- **Studio era windows.** Which production brands need explicit era buckets, and what windows are empirically supported by the catalog? TODO.
- **Multi-anchor similarity ("like Inception *and* Memento").** Current design assumes a single anchor movie. Naive vector averaging is geometrically biased (pulls toward dense clusters, can fall off the embedding manifold, drowns out distinct flavors of diverse inputs) and not a principled "semantic average." Better approach is likely per-anchor search with score aggregation across candidates (mean / min / reciprocal rank fusion) rather than a single averaged-vector query. Director and franchise axes also need a multi-anchor combination story (intersection vs union). TODO.
