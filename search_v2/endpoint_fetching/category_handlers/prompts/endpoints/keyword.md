# Endpoint: Keyword (Unified Classification)

## Purpose

Resolves the parts of the call's intent that map onto the closed UnifiedClassification registry — 259 canonical members across 21 families covering genre / sub-genre, story-engine archetypes, source material, cultural tradition, animation form, seasonal framing, narrative mechanics (endings, twists, POV tricks), viewer-response tags, and audience tier. Each finalized member dispatches onto its backing posting list (keyword, source-material, or concept-tag index); execution returns a per-movie hit count which the schema's `scoring_method` converts into a [0,1] score.

## What does NOT belong here

When decomposing `attributes` from the call's intent, ignore content from these out-of-scope domains rather than coercing it into a registry member:

- Named real entities (persons, characters, franchises, studios) → entity / franchise / studio endpoints.
- Structured numeric / factual attributes (release date, runtime, rating, country of origin, streaming availability, budget, box office, popularity, reception) → metadata endpoint.
- Free-form thematic / tonal / experiential qualifiers without a registry member ("cozy", "unsettling but not gory", "rainy-day melancholy") → semantic endpoint.
- Awards → award endpoint.

A sibling category handler in the same firing owns those facets. If nothing in the call's intent maps onto the registry, keyword has no clean fit — surface that honestly via empty `potential_keywords` on the walk attribute, or via `weaknesses` that names the gap. The bucket-level commitment is allowed to leave keyword unfired (single-endpoint buckets via `should_run_endpoint=false`; multi-endpoint buckets by simply omitting keyword from `coverage_assignments`). Do not coerce out-of-scope intent into a poor registry match.

## Where the keyword analysis lives

In single-endpoint buckets the analysis (`attributes` / `potential_keywords`) and the commitment (`finalized_keywords` / `scoring_method`) live together in one `KeywordQuerySpec`.

In multi-endpoint buckets (preferred-fallback, semantic+augmentation, audience-suitability) the analysis is hoisted to a bucket-level `keyword_walk` field that sits BEFORE the coverage_assignments commitment, while the commitment lives in a thin `keyword_parameters` slot AFTER it. Same content, just split in two so the LLM walks the registry concretely before the bucket decides whether to fire keyword at all. Refer to the schema descriptors for exact field locations.

## Classification registry

{{CLASSIFICATION_REGISTRY}}

When a concept could plausibly fit more than one family, compare candidate definitions directly — the definition that names the concept specifically wins over one that only covers it incidentally.

## Reading inputs as keyword facets

A single brief can carry multiple registry-relevant facets simultaneously. Decompose into one `attribute` per distinct facet — not per phrase. Typical combinations:

- Genre + cultural tradition: "scary Hindi films" → two attributes (horror feel; Hindi cinema tradition).
- Genre + source material: "biographical dramas" → two attributes (drama feel; biographical source).
- Sub-genre + ending shape: "slasher with a twist" → two attributes (slasher sub-form; twist mechanic).
- Combined member: when the registry has a single member that absorbs both facets (e.g., `TEEN_HORROR` for "teen horror," `HOLIDAY_ROMANCE` for "Christmas rom-com"), collapse to one attribute. The decomposition serves the registry, not the surface phrasing.

Out-of-scope content per the boundaries above is ignored at this step, even if it appears in the inputs.

## Surface forms and aliases

User phrasing often paraphrases canonical labels. Match by meaning, not literal echo:

- "Bollywood" → `HINDI` (the Hindi film tradition, not the audio track).
- "Biopic" → `BIOGRAPHY`.
- "Does the dog die?" / "animal death" → `ANIMAL_DEATH`.
- "Short films" / "shorts" → `SHORT` (form-factor classification, not runtime cutoff).
- "Twist ending" → `PLOT_TWIST`, unless the phrasing names a specific ending type.

These examples illustrate the principle; the right registry match is often definitionally clear even when the inputs do not literally echo the label. Do not force them mechanically.

## Authoring `strengths` and `weaknesses` per candidate

Each `potential_keywords` entry carries a registry member plus two short prose fields. They are walk-phase scaffolding: surface what the member would genuinely do at retrieval time given the parent attribute. The commitment phase reads them, but the gate from candidates to `finalized_keywords` is the superset test below — strengths/weaknesses describe the slice; they do not by themselves authorize a commit.

## Commitment: superset test

Fire keyword only when the keyword — or the ANY-mode union of keywords — is a true superset of the movies the user is asking for. A superset means: every movie that genuinely satisfies the user's attribute would carry at least one of the chosen keywords.

**Over-pull is acceptable.** The keywords also covering unrelated movies is not a failure of this test. Semantic refinement on the same call narrows the noise, and broadness in this trait is recovered by another trait's specificity.

**Gaps fail the test.** If a movie that genuinely satisfies the user's attribute could carry none of your chosen keywords, the set is not a superset. Firing will zero genuine matches under ADDITIVE-multiply. Abstain.

**Stretching intent fails the test.** If the keywords name something semantically adjacent to the user's attribute rather than the attribute itself, you are stretching. Firing will tag-match adjacent-but-irrelevant movies at 1.0 while genuinely relevant movies that lack those tags score 0. Abstain.

Apply the test once over the union of your finalized members in ANY mode. If the union passes the test, commit; if it fails on any of the three conditions, drop members until the remainder passes — or abstain entirely if no remaining subset passes. In multi-endpoint buckets, abstention means omitting keyword from `coverage_assignments`. In single-endpoint keyword buckets the schema requires at least one finalized member; the test still applies as guidance to pick the best registry fit, since those buckets routed here precisely because the user's attribute is registry-clean.

## Reading the brief for scoring_method

The scoring_method defaults to ANY. ALL is reserved for the case where the user named multiple distinct attributes that should compound, each independently demanded.

**Singular intent → ANY.** The user's expression names one attribute. The keyword commit may include multiple registry members because you have converted that one attribute into several registry surface forms — paraphrases, alternative routes, sub-form alternatives. Matching any one is sufficient evidence the user's one thing is present.

**Plural intent → ALL is on the table.** The user's expression names multiple distinct attributes the user wants present together — separate things, each independently demanded, compoundable. Each must be matched for the call's intent to be satisfied.

**Operational test:** read the call's expressions. One expression with multiple keywords commits to ANY. Multiple expressions naming genuinely distinct attributes that the user conjoined may commit to ALL.

When N=1 (one finalized keyword) the two modes are mathematically identical — default to ANY and move on.
