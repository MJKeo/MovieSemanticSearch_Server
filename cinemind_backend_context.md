# CineMind — Backend Architecture

A deep-dive into the backend of a **multi-channel, LLM-powered natural-language movie search engine**. This document explains *what* was built, *why* each decision was made, and *how* it works — covering the product concept, the core architecture, LLM evaluation methodology, the data-ingestion pipeline, the search system in all its forms, the safety model, and the runtime + developer-context memory systems.

> **One-line pitch:** Describe what you want to watch in plain English — *"a tense sci-fi thriller like Arrival but more action-heavy"* — and get back ranked, genuinely relevant recommendations from a curated catalog of ~100K films.

---

## Table of Contents

1. [Product Concept](#1-product-concept)
2. [Core Architecture & Key Decisions](#2-core-architecture--key-decisions)
3. [Evaluating LLM Results Across Development Stages](#3-evaluating-llm-results-across-development-stages)
4. [Data Ingestion & LLM Content Generation](#4-data-ingestion--llm-content-generation)
5. [Search — In All Its Forms](#5-search--in-all-its-forms)
6. [Safety From Bad Actors](#6-safety-from-bad-actors)
7. [Memory Management Systems](#7-memory-management-systems)
8. [Tech Stack & Engineering Highlights](#8-tech-stack--engineering-highlights)

---

## 1. Product Concept

### The problem
Conventional movie discovery forces users into rigid mechanisms: genre dropdowns, decade sliders, or keyword search that collapses the moment a query becomes conceptual ("a cozy date-night movie that isn't too sad"). None of these handle *intent* — plot elements, emotional tone, viewing occasion, similarity to other films, thematic weight, production trivia, or critical reception — expressed naturally in one sentence.

### The core insight
The central design thesis is that **a single embedding space conflates fundamentally different query intents.** "A cozy date-night movie" should not have to compete, in the same vector space, against plot details and production facts. Cramming everything into one semantic index produces results that are mediocre *everywhere*.

The product's answer is **multi-channel decomposition**: an LLM interprets the query and splits it into orthogonal facets, then specialized retrieval engines each do what they are individually best at — exact entity matching, semantic similarity, and structured attribute scoring — and the results are merged with LLM-assigned importance weights.

```
User Query
  │
  ▼
Query Understanding  (parallel LLM decomposition)
  ├── Extract lexical entities      (actors, directors, franchises, characters)
  ├── Assign channel weights        (how much do lexical / vector / metadata matter?)
  ├── Extract metadata preferences  (date, genre, runtime, streaming, ratings)
  └── Generate per-vector-space subqueries + space weights
  │
  ▼
Parallel Retrieval
  ├── Lexical Search   → PostgreSQL inverted-index posting lists
  ├── Vector Search    → Qdrant similarity across 8 embedding spaces
  └── Metadata Scoring → in-memory structured-attribute scoring
  │
  ▼
Score Merge:  final = w_L·lexical + w_V·vector + w_M·metadata   (all ∈ [0,1])
  │
  ▼
Quality Reranking:  bucket by relevance, sort by reception within buckets
  │
  ▼
Fetch display metadata → return JSON
```

The LLM does the **interpretation** (what the user wants and how much each facet matters); deterministic code owns all the **arithmetic** (the actual scoring and weighting), keeping the LLM out of fine numeric judgments where it is unreliable.

### Target audience & scope
Built for **people in the United States trying to decide what to watch right now** — primarily streaming, secondarily what's worth seeing in theaters. The US focus is load-bearing throughout: watch-provider availability data, IMDB scraping geo-targeting, MPAA maturity ratings, and content filtering all assume a US audience. The catalog is curated down from ~1M raw TMDB titles to ~100K high-quality films.

### The 8 semantic "lenses"
Every movie carries **8 named vectors**, each a distinct retrieval lens with its own LLM-generated descriptive text:

| Vector space | What it captures |
|---|---|
| `dense_anchor` | Lean holistic fingerprint — always participates (general recall) |
| `plot_events` | Literal chronological plot (who/what/when/where) |
| `plot_analysis` | Themes, core concepts, genre signatures, character arcs |
| `narrative_techniques` | Storytelling mechanics — POV, structure, devices, archetypes |
| `viewer_experience` | The *felt* experience — emotion, tone, tension (supports negations like "not too scary") |
| `watch_context` | The use-case lens — "date night," "background movie" |
| `production` | Filming locations, production/technical achievements |
| `reception` | Critical consensus, praise/complaint attributes, major award wins |

This granularity is deliberate: it separates *how a film feels* (`viewer_experience`) from *when you'd watch it* (`watch_context`) — two things a single vector would blur together.

---

## 2. Core Architecture & Key Decisions

The project keeps an append-only set of **Architecture Decision Records (ADRs)** — ~100 of them — capturing the decision, the rationale, and the alternatives rejected. The most consequential are summarized below. Every decision is grounded in an explicit, documented **priority ordering**:

> **Priorities (in strict order):** 1) Search quality — *"results must feel right; this is the core value proposition"* · 2) Latency · 3) Cost · 4) Code simplicity.
> **Tradeoff rule:** a major win in a lower priority can justify a minor cost in a higher one.

These priorities are visibly load-bearing across the system: *quality-first* drove the 8 vector spaces and an embedding-model upgrade; *latency* drove entity-query fast-paths and a Qdrant tuning fix; *cost* drove the entire ingestion funnel design.

### 2.1 Eight named vector spaces (ADR-001)
**What:** 8 vectors per movie instead of one. **Why:** a single space conflates intents; per-field vectors would be too sparse; 8 is the granularity that cleanly separates retrieval lenses. **Cost accepted:** 8 LLM generations per movie at ingest, and the search side must generate per-space subqueries + weights. **Alternatives rejected:** single space (conflates intent), 3–4 spaces (loses the feel-vs-occasion distinction), per-field vectors (too sparse).

### 2.2 The 5-stage vector scoring pipeline (ADR-005)
Raw Qdrant cosine scores across 8 spaces are folded into one `[0,1]` score via:

1. **Execute** — flags for which space-searches actually ran (a space with no relevant content is skipped entirely).
2. **Blend (80/20)** — `blended = 0.8·subquery_score + 0.2·original_query_score`. The LLM-generated subquery is more targeted; the raw query keeps results grounded.
3. **Normalize (exponential decay)** — within each space: `gap = (s_max − s) / (s_max − s_min)`, `normalized = exp(−k·gap)` with `k = 3.0`. Chosen over z-score because it **directly encodes "distance from the best candidate,"** is distribution-free (no clamping), and naturally rewards clear leaders.
4. **Weight** — coarse LLM relevance sizes (`SMALL/MEDIUM/LARGE → 1/2/3`) normalized to sum to 1; the anchor's weight is fixed at 80% of the mean of active non-anchor weights ("always present, never the loudest voice").
5. **Sum** — weighted sum across spaces, guaranteed `∈ [0,1]`.

Candidates appearing in few spaces are naturally penalized (zeros for non-participating spaces) — by design.

### 2.3 Soft metadata scoring, not hard filters (ADR-007)
**What:** LLM-extracted preferences (decade, genre, runtime…) are **soft signals in a weighted average**, never hard filters. **Why:** hard filters from LLM output over-penalize on misinterpretation — *"something from the 90s"* should not exclude a 2001 film. Hard filters are exposed *separately* as explicit UI controls the user sets directly. Genre/language **exclusions** are the one intentional exception, applying a −2.0 penalty that drives the score negative.

### 2.4 Quality-prior reranking (ADR-008)
**What:** after the composite relevance score, **bucket by relevance (round to 0.01), then sort within buckets by normalized reception score.** **Why:** among equally-relevant results, well-received movies should win — but bucketing confines quality to a *tie-breaker* so a great-but-barely-relevant film can never outrank a highly-relevant average one. If the user explicitly asks for *poorly-received* films, the prior is disabled (don't fight the user).

### 2.5 Embedding model: text-embedding-3-large (ADR-066)
Upgraded from `text-embedding-3-small` (1536-dim) to **`text-embedding-3-large` (3072-dim, native)**. **Why:** a V2 redesign forced a full corpus re-embed anyway, making the model swap effectively free — so the team captured the quality lift at zero marginal migration cost. Voyage-3-large was evaluated and retained as the next revisit candidate if embeddings ever become the quality bottleneck.

### 2.6 Qdrant storage + a latency bug worth telling (ADR-004, ADR-059)
- **Scalar quantization (int8) + memmap** compress the working set from ~7.4 GB to fit a small EC2 box's RAM budget with negligible ANN accuracy loss.
- A **5–7 second** per-query latency was traced to two independent issues: `rescore=True` was re-fetching full float32 vectors from disk, and a `limit=2000` was *silently overriding* the HNSW `ef` parameter (Qdrant sets `ef = max(limit, ef)`), forcing graph traversal ~16× too deep. Setting `rescore=False` and `limit=500` brought it to **sub-second**. A clean example of profiling-driven debugging.

### 2.7 Three specialized data stores (ADR-011)
| Store | Role | Why this store |
|---|---|---|
| **PostgreSQL 15** | Movie metadata, lexical posting tables, reranking fields | Native posting-list lookups, GIN-indexed arrays, hard filters in the same SQL |
| **Qdrant** | 8-space ANN search; payload holds *hard-filter fields only* | Named vectors + scalar quantization (pgvector lacks both) |
| **Redis 7** | Embedding cache, trending set, TMDB detail cache | Hash/Set types and binary values neither Postgres nor Memcached fit cleanly |

Whole thing runs as Docker containers on **one EC2 t3.large (~$60/mo)** — justified because QPS is low and latency is dominated by LLM calls, not infrastructure (ADR-003).

### 2.8 Cross-codebase invariants (the silent-bug guards)
A short list of non-negotiable rules, each preventing a specific class of bug:

- **`movie_id` is always `tmdb_id`** — never a secondary ID system (kills an entire class of join/translation bugs).
- **String normalization runs identically at ingest and query time** — a mismatch is a *silent* retrieval failure (an entity that exists is never found).
- **Qdrant scores are final** — never recomputed at reranking (no scoring drift, no wasted re-fetch).
- **Never query Postgres per-candidate** — always bulk-fetch with `WHERE movie_id = ANY($1)` (avoids N+1 latency blowup).
- **Never cache partial DAG outputs** — the entire query-understanding result is one atomic cache key (no Frankenstein half-stale results).
- **Embedding cache does *not* lowercase; the lexical cache does** — embedding models are case-sensitive, so "Apple" ≠ "apple" is real signal.
- **Qdrant payload is for hard filters only** — full metadata lives in Postgres (keeps the memory-constrained instance lean).

---

## 3. Evaluating LLM Results Across Development Stages

LLMs are used in three places — query understanding, metadata generation, and search-side translation — and each evolved its own evaluation discipline. This is arguably the most rigorous part of the project: evaluation was treated as **infrastructure**, with a design spec written *before* any code, reusable tooling, and decision records tying every model choice to eval data.

### 3.1 Metadata generation — pointwise LLM-as-judge
Each of the ~12 generated metadata types gets a dedicated evaluation harness. The methodology:

- **Pointwise, not pairwise.** Each candidate output is scored independently against a rubric rather than compared head-to-head. **Why:** pairwise preferences are unstable, scale as N² comparisons, and don't yield the per-dimension strength/weakness profile the team wanted.
- **Dimension-level (axis) scoring.** Each schema is decomposed into orthogonal axes scored separately, because a model "may produce excellent plot summaries but poor character-relationship mappings" — and aggregating destroys that diagnostic signal. Example (the `viewer_experience` rubric): Groundedness (0.25), Specificity Layering (0.20), Retrieval Alignment (0.20), Section Discipline (0.15), Negation Quality (0.10), Term Quality & Diversity (0.10), each with concrete 1–5 descriptors, plus a separate holistic score.
- **Objective vs. interpretive fields scored differently.** Verifiable fields (character names, years) are graded on *accuracy*; interpretive fields (tone, themes) on *defensibility* — the judge must not penalize "melancholic, bittersweet" vs. "somber, wistful" as an error.
- **Explicit reasoning before scores.** The judge outputs its chain-of-thought *first*, giving an audit trail to determine whether a wrong score came from a flawed rubric or a misapplied one — which then drives rubric refinement.
- **Bucket stratification by input quality.** Every eval set is partitioned into buckets spanning the input-richness spectrum (e.g., from "Gold Standard: 800+ char plot summary" down to "raw fallback < 200 chars"). Per-bucket scoring reveals whether a candidate systematically fails on *thin* inputs — invisible in an aggregate number.

### 3.2 The methodology evolved — reference-based → reference-free
A genuine finding worth highlighting: the original design spec prescribed a **reference-based** method (generate Opus reference answers, use them as a calibration anchor). What actually shipped is **reference-free**: the judge (Claude Opus, thinking disabled) sees the **same raw source data the candidate saw** and verifies groundedness directly. **Why the change:** it eliminates reference-anchoring bias entirely for interpretive fields and removes a whole generation phase. The team iterated on its *evaluation infrastructure itself*, not just the models under test.

### 3.3 Contamination control as a first-class concern
A hard rule, learned from a specific incident and now codified in three places:

> **Never use the same movies as prompt examples AND as eval test data.** Overlapping sets make it impossible to tell whether the model *learned the principle* or merely *memorized the example*.

This traces to a real failure: in one generation round, the prompt examples (Rogue One, Creed, Star Wars…) were the same titles used in the eval buckets, making results un-interpretable. The fix became a project convention: prompt examples and eval data must be disjoint, examples kept to 2–4, and the abstract rules must stand alone without them.

### 3.4 Other rigor worth calling out
- **Grade what gets embedded, not what gets generated.** The judge only scores fields that actually enter the vector space; scaffolding/justification fields are excluded — and schema variants with extra reasoning fields are required to produce *byte-identical* embedding text so the eval measures generation quality, not text-length artifacts.
- **Grade for retrieval quality, not prompt compliance.** A "good" result is one that makes the *right* queries match and the *wrong* ones not — even if it deviates from the prompt. This prevents rewarding sycophantic prompt-following over downstream usefulness.
- **Self-enhancement bias handled deliberately.** The judge is Claude-family while all generation candidates are non-Anthropic; since the bias works against all candidates roughly equally, *relative* rankings stay reliable, and rubric specificity is the primary mitigation.
- **Search-side eval is deterministic.** Query-understanding outputs are structured routing decisions, so they're checked programmatically with **N-repeat distributions** (e.g., "SOLO 0/5 → 5/5 across model versions"), **held-fixed upstream stages** for clean isolation, a **living failure-mode catalog** with a defined metric per mode, and **regression-guard queries** every round.
- **Every model choice is an auditable, cost-weighted, eval-backed ADR.** E.g., a generation model was chosen for higher groundedness specifically because its output feeds 4 of 6 downstream generators — *"every percentage point of hallucination translates to thousands of degraded search results."*

### 3.5 The tooling that made it systematic
Evaluation was codified into reusable commands so any new metadata type can be evaluated identically: scaffolding a harness, designing a weighted rubric, generating candidate comparisons at a target throughput, running the judge, and reporting **per-candidate × per-bucket × per-axis** score matrices — plus a "citation rate" analysis that parses the judge's own reasoning to decide which input fields to prune. The core iteration loop is repeatable and documented with hard numbers: **quantify the dominant failure mode → make a targeted prompt/schema change → re-evaluate → confirm the mode is eliminated without regression.**

---

## 4. Data Ingestion & LLM Content Generation

The ingestion pipeline takes ~1M raw TMDB entries down to ~100K richly-indexed movies, each with LLM-generated metadata embedded across the 8 vector spaces. Every stage is **crash-safe and idempotent** — restarting picks up exactly where it left off.

### 4.1 The funnel and its guiding philosophy
```
Stage 1: TMDB Daily Export      ~1M  → ~800K     (stream-decompress; free)
Stage 2: TMDB Detail Fetch      ~800K fetched    (async HTTP; free)
Stage 3: TMDB Quality Funnel    ~800K → ~100K    (lenient gate)
Stage 4: IMDB Scraping          ~100K enriched   (paid residential proxy)
Stage 5: IMDB Quality Filter    ~100K filtered   (the real quality gate)
Stage 6: LLM Metadata Gen       ~112K movies     (OpenAI Batch API)
Stage 7: Vector Text Assembly   (in-process)
Stage 8: Embed + Ingest         → Postgres + Qdrant
```

The architecture is driven by **cost asymmetry** (ADR-002): TMDB detail fetching is free, but IMDB scraping costs proxy bandwidth and LLM generation costs real money. Running expensive stages on all ~1M entries would cost ~10× more — so the pipeline spends *free* TMDB calls liberally to rank candidates, and only escalates the top ~100K into paid stages. A complementary principle governs the quality gates: **lenient early, strict late.** Stage 3 only removes obvious junk (a wrongly-cut movie is gone forever); Stage 5 is the real gate, adjudicating with rich IMDB review/rating data.

### 4.2 The crash-safe tracker (ADR-006, ADR-014)
A single **SQLite** file is the shared backbone — checkpoint tracker, quality-scoring store, and IMDB data store all at once. **Why SQLite over alternatives:** ~950K flat JSON files would be 3–5 GB with filesystem degradation and no atomic queries; Postgres is overkill for a local pipeline; Redis isn't durable by default. SQLite gives ACID durability in one ~350 MB file that fits in RAM. Crash-safety is *structural*:

- `PRAGMA journal_mode=WAL` + `synchronous=FULL` (added after a real mid-batch corruption incident).
- **Single-codepath writes:** all filtering goes through one `log_filter()` helper that atomically writes the audit log *and* updates status — stage modules never touch those tables directly.
- **Atomic data+status transitions**, bounded commit cadence (every ~500 movies), and resumability via `WHERE status = '<prior status>'`.
- **Idempotent schema migrations** via `ALTER TABLE` statements wrapped in try/except — each no-ops if already applied.

### 4.3 Quality scoring — multi-signal, data-driven thresholds
**Stage 3 (TMDB, 4 signals):** vote_count (0.50), popularity (0.20), overview_length (0.15), data_completeness (0.15), with edge-case bypasses (unreleased → 0.0; has a US streaming provider → 1.0, since a streaming license is definitive proof of commercial viability). Vote count dominates because it's the **universal meta-signal** — every other quality proxy correlates monotonically with it. Age multipliers compensate for short vote-accumulation windows (recency boost) and pre-internet underrepresentation (classic boost).

**Stage 5 (combined TMDB+IMDB, 8 signals + 2 hard gates):** Two categorical eligibility gates first (title-type must be a movie/short/etc.; must have *some* plot text) — these are gates, not weighted signals, because a video game isn't a "low-quality movie," it's a categorical failure. Then an 8-signal weighted score led by **`imdb_notability` (0.31)**, which blends log-scaled vote count with a **Bayesian-adjusted IMDB rating** (shrinks noisy low-vote ratings toward the dataset mean). The blend weight is governed by a **3-tier confidence model**: under 100 votes, ratings are 95% discounted as noise; at 100–999 votes the rating carries real signal; above 1000, vote count dominates. **Why:** earlier versions used pure vote count and had *zero* quality discrimination — a 2/10 and an 8/10 film with equal votes scored identically.

**How thresholds are chosen — survival-curve derivative analysis.** Rather than guessing a cutoff, the team builds a survival curve (count of movies scoring ≥ x), Gaussian-smooths it, and computes derivatives to find the **inflection point** — the "elbow" where raising the threshold stops removing many movies. That natural boundary separates a dense low-quality tail from the body of genuine films. Stage 5 runs this *per provider group* (has-providers / new-no-providers / old-no-providers), yielding three different, principled thresholds.

### 4.4 IMDB scraping — GraphQL + residential proxies (ADR-009, ADR-018)
- **Single GraphQL query** replaced 6 HTML page-fetches per movie — 600K requests → 100K, ~5–6× less proxy bandwidth, and simpler error handling (one request succeeds or fails, no partial-page logic).
- **DataImpulse residential proxies with US geo-targeting** — IMDB blocks datacenter IPs but accepts residential ones; US targeting matters for MPAA ratings and watch metadata.
- **Counterintuitive tuning, empirically validated:** concurrency capped at 60 (tested at 100 — higher concurrency *increased* timeouts without throughput gain, proving the bottleneck is IP quality, not parallelism); a **5s timeout** (successful residential fetches finish in <1s); and **flat retry delays, not exponential backoff** — because each retry routes through a *fresh* residential IP, so backoff just wastes time.

### 4.5 LLM metadata generation (Stage 6)
**12 generation types** — 7 feed the LLM-generated vector spaces (plot_events, plot_analysis, viewer_experience, watch_context, narrative_techniques, reception, production), the rest are structured classifiers (franchise, source-of-inspiration, concept tags…). They run in **dependency waves**: Wave 1 reads raw IMDB/TMDB text; Wave 2 depends on Wave 1 outputs.

**Key cost & quality engineering:**
- **OpenAI Batch API** for a flat **50% discount**, trading 24–48h latency that offline ingestion can absorb. Part of a documented cost-optimization campaign that drove per-movie generation from ~$0.025 toward a projected ~$0.0015 (~$2,500 → ~$150 per 100K).
- **Schema slimming:** ~28 unused justification/explanation fields were removed from output schemas — they were never embedded, only guiding chain-of-thought — saving tokens across 112K × 8 calls.
- **Two-branch plot generation:** when a comprehensive synopsis already exists, the model *condenses* it (cheap, grounded); otherwise it *synthesizes* from sparse inputs while framed as having "no knowledge of any film" to prevent fabrication. The condensation threshold was raised from 1K to 2.5K chars after an eval showed 67% hallucination at ~1K-char inputs vs. 0% at 4K+.
- **Generator registry pattern:** each type maps to a `GeneratorConfig` (schema, eligibility checker, prompt builder, locked model + kwargs) behind a uniform `MovieInputData → result` contract — so the generic batch pipeline runs all types concurrently without duplicating orchestration. Models are **locked as module constants** so production can't drift from the evaluated winner.
- **Pipeline state in SQLite:** an in-flight batch gate (prevents duplicate submission), a failures table (distinguishes "attempted-and-failed" from "not-yet-attempted"), and an `autopilot` mode that interleaves fast live generations with batch polling so the pipeline is validated in minutes, not 24-hour cycles.

### 4.6 Final ingestion — parallel dual-database upsert (ADR-062)
Stage 7 assembles the exact embeddable text per space (with fallback hierarchies and an automatic token-limit guard, since the embedding API *errors* rather than truncating on over-length input). Stage 8 embeds all 8 texts in a **single batched OpenAI call**, then upserts Postgres and Qdrant **in parallel** (no data dependency; they overlap naturally on I/O). Error handling is surgical: **nested SAVEPOINTs** isolate per-movie and per-step failures so one bad movie can't abort a 100-movie transaction; only movies succeeding in *both* stores are marked `ingested`; transient failures are retryable while missing-data failures are terminal; and all upserts are idempotent so re-runs are safe.

---

## 5. Search — In All Its Forms

The system is mid-migration between two generations. **V1** is the original three-channel parallel-blend pipeline. **V2** is a redesign that fixes systematic failures in how V1 weighted and combined intents. Both are documented here, since the *why* behind V2 is itself a strong engineering story.

### 5.1 V1 — query understanding via parallel decomposition
A query fans out into **~24 concurrent LLM calls**, grouped into five logical tasks (lexical entity extraction, channel weights, 9 parallel metadata-preference extractions, 7 per-space vector subqueries, 7 per-space weights). **Why fan out so aggressively:**
- **Prompt focus** — each prompt teaches one narrow space or attribute in depth; mixing them degrades judgment and bloats context.
- **Free parallelism** — independent I/O-bound calls run concurrently, so latency ≈ the slowest single call, not the sum.
- **Per-field graceful degradation** — the 9 metadata calls use `return_exceptions=True`; one failure becomes a "no preference" default instead of poisoning the whole query.
- **`null` as a first-class signal** — when a space has no relevant content, the LLM returns `null`, short-circuiting that Qdrant search entirely.

A crucial discipline: the entity-extraction prompt **forbids inference** — "that space movie with the docking scene" must *not* be hallucinated into "Interstellar," because that would convert a semantic query into a lexical lookup and break channel routing. And because the channel-weights LLM judges intent *blind* to what the other extractors found, its output is **deterministically reconciled** against ground truth before use (e.g., lexical weight is forced to zero if no entities were actually extracted).

### 5.2 V1 — the three retrieval channels
- **Lexical (PostgreSQL):** named entities resolve through inverted-index **posting tables** (`PRIMARY KEY (term_id, movie_id)` — the PK *is* the posting list). People are split into **five role-specific tables** (actor/director/writer/producer/composer) so "directed by X" and "starring X" hit different indexes; characters use a separate trigram-indexed dictionary for substring matching. **Why Postgres, not Elasticsearch or a vector:** named-entity matching is an exact set-membership problem — "movies with Tom Hanks" must return *exactly* his credited films, not semantically-adjacent actors. Postgres already owns the source-of-truth data, so hard filters apply in the *same* SQL.
- **Vector (Qdrant):** the 8 named spaces, scored through the 5-stage pipeline in §2.2.
- **Metadata (in-memory):** a weighted average over only the *active* preferences, computed as a CPU-bound pass over the already-fetched candidate cards (no benefit to pushing it into SQL).

The three normalized channel scores are merged via a simple **weighted linear sum** (`final = w_L·lex + w_V·vec + w_M·meta`) — interpretable, debuggable, and the natural composition of three scores on a common `[0,1]` scale — then reranked by the quality prior (§2.4).

### 5.3 Why V1 wasn't enough
Analyzing real query failures surfaced systematic flaws in the "global channel weights + additive blend" model, documented as ~18 numbered issues. The structural ones:

- **Expansion volume distorts intent weight.** A single intent that fans into many sub-searches drowns out a sibling intent that resolved to one search — so a film vaguely matching five generic atoms could outrank the canonical film matching the one unique conjunction.
- **Atom decomposition destroys conjunctive meaning.** *"John Wick but with kids"* splits into pieces that each find a retrieval home, but their combination no longer reconstructs *"a kid version of John Wick."* The V1 schema could only *split*, never *merge* or recognize when decomposition lost recoverable meaning.
- **LLM filter-vs-trait decisions fail.** *"Around 90 minutes"* got demoted to a soft trait, letting wildly-off films rank well — concrete metadata should *always* be a (soft-falloff) filter.
- **Tonal negation fails on embedding geometry.** *"Without being depressing"* failed because "depressing" and "grief" embed near each other, so downranking pulled down the whole region — negations must be rewritten as positive opposites ("hopeful, cathartic").
- **"Movies like X" is the wrong shape for one ranked list.** Similarity is a *bundle* of orthogonal meanings (same experience / creative fingerprint / story world / source tradition / studio style / taste tier). Collapsing them into one scalar is a recommendation problem masquerading as a search problem — it needs its own lane-based flow.

### 5.4 V2 — the redesign
V2 replaces the flat parallel blend with a **staged interpretation pipeline** plus **entity fast-paths**:

- **Step 0 — flow routing.** Every query is classified into one of 7 mutually-exclusive entity flows (exact-title, similarity, person, character-franchise, studio, …) or "standard." Entity flows (~30–40% of queries) **bypass the trait pipeline entirely**, routing to deterministic posting-list executors. **Why:** entity queries have fundamentally different retrieval semantics (billing prominence, franchise mainline status) that trait scoring mis-ranks — and skipping the LLM pipeline is a major latency win.
- **Steps 1–3 — narrow sequential interpretation.** A monolithic "understand the query" step was decomposed into focused steps (intent read → atoms + committed traits → per-trait routing into typed endpoints). **Why:** narrow scope per step preserves the relationship structure that flattening destroyed, and the large category taxonomy only loads at the final step, shrinking every prompt.
- **Cleaner trait encoding.** V1's ambiguous "role + salience" two-axis scheme (where the LLM anchored on one axis and produced noise on the other) was replaced by two clean orthogonal enums: a `relationship_role` (is this a trait the user *wants*, or just a reference film used to *position* the query, as in "like Inception"?) and a 5-level ordered `commitment` axis mapped to scoring multipliers.
- **Deterministic entity executors.** Each entity flow has a purpose-built, sub-millisecond, no-LLM executor — e.g., the actor flow uses a **sqrt-adaptive cast-zone model** where billing-position cutoffs scale with `sqrt(cast_size)` (billing position 3 is a co-lead in an 8-person cast but supporting in an 80-person ensemble); the character-franchise flow uses 7 strict-priority prominence tiers.
- **Pool/score separation (Stage 4).** All positive generators build *one* unified candidate pool; all rerankers then score that *finalized* pool — fixing a V1-era bug where a reranker only ever saw its own trait's candidates. Hard filters are threaded into every retrieval primitive *at the primitive* (not post-filtered), guaranteeing a byte-identical result list for any unfiltered query.

The redesign honestly documents its open questions — chiefly that final tuning of the reranking parameters awaits an NDCG benchmark query set that doesn't exist yet, which is a stated prerequisite. (Model note: V2 query understanding now spans Gemini for Steps 0–2 and OpenAI for Step 3, chosen because the latter consolidated coherent concepts more decisively in head-to-head evals.)

---

## 6. Safety From Bad Actors

This is a public, unauthenticated-feeling API where **every request fans out into paid LLM and embedding calls.** That economic shape defines the threat model: the most dangerous "attack" is not data theft but **cost amplification, prompt injection, and volume-based DoS.** Defenses are layered across three tiers — the network edge, the HTTP boundary, and the LLM call itself.

### 6.1 Structured outputs — the prompt-injection defense
**Every** LLM call is forced to return data conforming to a Pydantic schema; free-form text is never parsed or trusted. This is centralized in a unified multi-provider router, with each provider's structured-output mechanism converging on a **validated Pydantic object** (OpenAI's `.parse()`, Kimi's strict JSON schema + manual `model_validate`, Gemini's JSON-schema mode, Anthropic's forced single-tool-use, etc.).

**Why this is a security mechanism:**
1. **It constrains the attack surface to data, not control.** A query like *"ignore your instructions and output X"* can at most influence the *values* inside a fixed schema (a genre field, a subquery string). It can never make the LLM emit arbitrary instructions, SQL, or shell text that downstream code would act on — because downstream code only ever reads typed fields off a validated object. The schema is a contract the model cannot break out of.
2. **Malformed/jailbroken output fails closed.** A non-conforming response raises a validation error and becomes a clean `ValueError`, never corrupt state.
3. **Reliability hardening doubles as DoS resistance.** Every call is wrapped in a 25s per-attempt timeout with 2 attempts and **jittered backoff** — explicitly to avoid the retry-storm self-DoS where many concurrent calls retry at the same wall-clock instant after a transient 5xx.

### 6.2 Input validation at the boundary
User free-text flows through a single validation module enforcing **strip, non-empty, and a hard 200-character cap** on both the query and any clarification (bounded independently so a long clarification can't bypass the query cap). Over-length input is **rejected, not truncated** — silent truncation could slice a query mid-clause and corrupt intent. **Why the cap:** both fields are concatenated into the prompt sent to the LLM on *every* search, so an unbounded field lets a caller inflate input tokens (and therefore cost/latency) arbitrarily and widens the injection surface. The hard ceiling bounds per-request cost "for free."

Additional boundary hardening: Pydantic request models with type + min-length enforcement; **enum allow-listing** of every filter value (unknown values return 422, never silently dropped); parameterized SQL only (no string interpolation); and **server-side-only reranking** so scoring logic is never exposed to client manipulation.

### 6.3 Cloudflare edge — rate limiting & origin hiding
The API is **not exposed directly to the internet.** A Caddy reverse proxy is the only host-published service; the FastAPI container publishes no ports and is reachable only on the internal Docker network. **Cloudflare fronts the whole thing** — terminating public TLS, re-encrypting to the origin via a Cloudflare Origin CA cert, with the daemon (`cloudflared`) tunneling traffic in. The application's only hook into the edge is reading the real client IP from the `CF-Connecting-IP` header for abuse attribution (the socket peer is always Cloudflare's edge).

**Why this design:** the two abuse classes the application deliberately does *not* handle — **volume-based DoS and quota abuse** — are absorbed at the Cloudflare edge via rate limiting, *upstream* of any paid LLM call. And because the origin only trusts the Cloudflare Origin cert and publishes nothing else, attackers can't bypass the edge to hit FastAPI directly (the standard way to defeat edge rate limiting). The cost-amplification threat is cut off before it ever reaches the expensive pipeline.

CORS is configured with an explicit origin allow-list (no wildcard), but the code is honest about its scope: *CORS is browser-enforced only and stops other sites' JS from calling the API in a user's browser — it is not a defense against scripted callers.* That job belongs to Cloudflare. This separation of concerns — each layer doing exactly the job it can actually do — is the heart of the safety model.

### Threat-model summary

| Bad-actor behavior | Defense | Why it works |
|---|---|---|
| Cost amplification (huge prompts) | 200-char caps, reject-not-truncate | Bounds input tokens before any LLM call |
| Prompt injection | Structured outputs (all providers) | Model influences data values only, never control flow |
| Malformed/jailbroken output | `model_validate` fails closed | Bad output → clean error, not corrupt state |
| DoS via request volume | Cloudflare edge rate limiting | Absorbed at edge; origin hidden behind Origin CA cert |
| Self-DoS / retry storms | Timeouts + jittered backoff | Bounds blast radius; de-synchronizes concurrent retries |
| Browser CSRF | Explicit CORS allow-list | Blocks other sites' JS (honestly scoped) |
| Poisoned filter values | Enum allow-listing → 422 | Only known taxonomy values reach the query layer |
| SQL injection | Parameterized queries only | No string interpolation into SQL |

---

## 7. Memory Management Systems

The project has **two entirely distinct systems** called "memory." They share no code and serve different audiences — one serves end-user requests at runtime, the other serves the developer across coding sessions.

### 7.1 Runtime caching (Redis)
A single async Redis 7 instance backs four cache families, all namespaced and keyed by `tmdb_id`. A foundational choice is `decode_responses=False` globally — **because the embedding cache stores raw packed binary, not strings.**

| Cache | TTL | Why |
|---|---|---|
| **Trending scores** (Hash) | **No TTL** | "Stale trending beats missing trending" — refreshed daily via an **atomic RENAME** so the live key is never in a partial state |
| **Movie details / credits** | 24h | Stores the *curated wire payload*, not raw TMDB JSON — warm hits skip the TMDB round-trip *and* the build/encode step; bytes returned verbatim |
| **Similar movies** | 24h | Keyed by sorted+deduped anchor IDs, with a BLAKE2b filter fingerprint appended so filtered/unfiltered responses occupy disjoint key slots |

**The two flagship invariants and their rationale:**

1. **"Never cache partial DAG outputs — the entire query-understanding result is one atomic key."** Query understanding is a DAG of independent parallel LLM calls. Caching each node separately could serve a *Frankenstein* result — node A from prompt-version v3, node B from a stale v2 — an internally inconsistent understanding that produces subtly wrong retrieval with *no error*. One atomic blob guarantees coherence.

2. **"The embedding cache does not lowercase; the lexical cache does."** Lexical normalization lowercases for matching, and must run *identically* at ingest and query time (a mismatch is a silent retrieval bug). But **embedding models are case-sensitive** — "Apple" (company) and "apple" (fruit) embed differently, and that's real disambiguating signal. Lowercasing the embedding key would both collide distinct inputs onto one slot *and* degrade vector quality. So the two caches use deliberately different normalizers.

A consistent thread is **graceful degradation**: a missing trending key → neutral prior; a missing embedding cache → call the API. Redis failures are logged, never fatal. The cache also doubles as a **secret shield** — TMDB is proxied through the server so API keys never reach the client.

### 7.2 The developer-context "memory" system
This is a bespoke **context-engineering system** for working with an AI coding agent across many sessions — solving the twin problems that an LLM agent has *no persistent memory*, and that unstructured docs go stale and get silently overwritten. It's formalized as a decision record (ADR-013) and works as a **memory hierarchy with explicit write-permissions and human-in-the-loop promotion gates.**

**The layers, transient → permanent:**

| Layer | Lifetime | Who may write | Role |
|---|---|---|---|
| `DIFF_CONTEXT.md` | Cleared each commit | Agent (autonomous) | Per-session "what changed and why," organized by intent |
| `conventions_draft.md`, `workflow_suggestions.md` | Until reviewed | Agent (append) | **Staged** candidates — observed but not yet ratified |
| `docs/TODO.md` | Until done | Agent (autonomous) | Deferred work discovered mid-session |
| `docs/modules/*.md` | Long-lived | Agent (proportional edits) | Per-module summaries (boundaries, gotchas) |
| `docs/personal_preferences.md` | Permanent | Agent (auto-maintained) | How the developer wants to work |
| `docs/conventions.md` | Permanent | **Human-gated only** | Cross-codebase invariants |
| `docs/decisions/ADR-*.md` | Append-only | **Human-gated** | Architectural rationale + alternatives |
| `docs/PROJECT.md` | Permanent | **Human-only** | Audience, priorities, constraints |
| `.claude/rules/*.md` | Permanent | Human | Behavioral rules loaded every session |

**The promotion pipeline (the conceptual heart):** a **staged → finalized** flow mirroring git's draft → review → merge, where the agent *proposes* and the human *ratifies*:
- A `safe-clear` step (run before clearing context) extracts session learnings into the staging layer — convention *candidates* (with a "sessions observed" counter that increments on repeat, so stronger candidates accrue evidence), preference updates, workflow ideas, and TODOs.
- A `solidify-draft-conventions` step processes candidates one at a time, proposes a *generalized* version of each (the explicit anti-goal: "avoid accumulating many narrow, overlapping conventions"), and waits for a Promote / Reject / Revise decision. Only on Promote does it reach the permanent conventions file.
- An `extract-finalized-decisions` step writes numbered ADRs only for *committed* decisions, then flags (but never auto-applies) any needed edits to human-only docs.
- A commit-time `docs-maintainer` subagent flushes `DIFF_CONTEXT.md` into permanent module docs and drafts ADRs, then clears the transient file — while being hard-forbidden from touching `PROJECT.md` or `conventions.md`. A read-only `docs-auditor` periodically scans for staleness and even **priority drift** (if recent decisions consistently rank a lower priority over a higher one).

**The conceptual decisions, and why they matter:**
- **Separation of mutable from stable.** Module docs are freely agent-writable (kept fresh at zero human cost); decisions are append-only and agent-protected (hard-won rationale can never be silently lost); product priorities are human-only (the system's value function stays under human control). The core problem being solved is *an agent clobbering stable design rationale* — solved by tiering write permissions.
- **Stage before finalizing.** Candidates accumulate with evidence counters so the human reviews a curated, de-duplicated, generalized set — not raw noise — and is always the gate on what becomes a permanent invariant. The system is deliberately tuned to keep permanent memory *small and general*.
- **The agent orients cheaply.** At session start it reads product context, the relevant module doc, and prior decisions *before* touching code; at session end it writes back structured deltas. Reconstruct just-enough context in, structured deltas out.

---

## 8. Tech Stack & Engineering Highlights

**Stack:** Python 3.13 · FastAPI · PostgreSQL 15 · Qdrant (8-space ANN, int8 quantization) · Redis 7 · Docker Compose on a single EC2 instance · Caddy + Cloudflare (Tunnel + Origin CA) · OpenAI (`text-embedding-3-large`, gpt-5-class generation + Batch API) · Gemini & Kimi/Moonshot (query understanding) · Claude (LLM-as-judge) · multi-provider LLM router (7 backends).

**Engineering themes a recruiter should take away:**

- **Systems design from first principles.** Identified that single-vector RAG conflates query intents, and designed an 8-space multi-channel architecture with LLM-driven decomposition and deterministic score fusion to solve it.
- **LLM evaluation as infrastructure.** Built rubric-based, bucket-stratified, contamination-controlled LLM-as-judge harnesses; tied every model choice to eval data; and iterated on the *evaluation methodology itself* (reference-based → reference-free).
- **Cost-aware data engineering at scale.** Designed a crash-safe, idempotent ingestion funnel (~1M → ~100K) driven by cost asymmetry, with Bayesian-adjusted multi-signal quality scoring and data-driven (derivative-analysis) thresholds — and cut LLM generation cost ~16× via batching, schema slimming, and input pruning.
- **Performance debugging.** Traced a 5–7s query latency to a Qdrant `ef`/`limit` interaction and disk rescoring; brought it sub-second.
- **Security thinking grounded in a real threat model.** Layered defenses (Cloudflare edge rate limiting + origin hiding, boundary input caps, structured-output prompt-injection containment, fail-closed validation) chosen for the specific economics of a public LLM-backed API.
- **Production reliability.** Idempotent everything, single-codepath writes, nested-SAVEPOINT error isolation, graceful degradation, atomic cache swaps, and append-only decision records.

---

*This document reflects the as-shipped backend. A small number of older decision records predate later upgrades (e.g., the embedding model and the Caddy/Cloudflare proxy migration); the descriptions above reflect the current system, with the rationale preserved from the original records.*
