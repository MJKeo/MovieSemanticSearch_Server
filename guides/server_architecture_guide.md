## Server Architecture Guide (v3)

This doc describes the end-to-end search system: request handling, query understanding, retrieval (lexical + vector), reranking, caching, movie-detail fetching, and the supporting data stores/jobs. Target audience: a new engineer ramping onto the project.

---

# 1) High-level components

All services run as Docker containers on a **single EC2 t3.large instance** (2 vCPU, 8 GB RAM), orchestrated with Docker Compose. Nginx runs on the same instance as a reverse proxy, handling HTTPS termination. There are no managed database services — Postgres, Redis, and Qdrant are all self-hosted containers.

**Why a single instance:** QPS is low and latency is dominated by LLM calls, not infrastructure. Managed services (RDS, ElastiCache, ECS) add $150–200/mo with no meaningful benefit at this scale. Everything fits comfortably within the t3.large RAM budget with headroom to spare.

---

### A) Search API Service (stateless "modular monolith")

Single Python backend service running in a Docker container.

* Runs the query-understanding DAG (multiple LLM calls)
* Kicks off lexical + vector search as soon as dependencies are ready
* Merges candidates, enriches minimal metadata, reranks, returns top N
* Proxies movie detail fetches from TMDB with a small Redis cache (1 day)

---

### B) Postgres (self-hosted Docker container)

Used for:

1. **Lexical search** (lexical_entities schema)
2. **Canonical minimal movie metadata** for card rendering + reranking

No managed RDS. A daily `pg_dump` ships to S3 for backup.

---

### C) Qdrant (self-hosted Docker container, quantized)

Holds:

* One "point" per movie (`point_id = movie_id`)
* **8 named vectors** per movie (anchor, plot_events, plot_analysis, narrative_techniques, viewer_experience, watch_context, production, reception), each 1536 dimensions
* Payload fields required for **hard filters**: `release_ts`, `runtime_minutes`, `maturity_rank`, `genre_ids`, `watch_offer_keys`

**Memory management — scalar quantization + memmap:**

Without compression, 150K movies × 8 vectors × 1536 dims × 4 bytes (float32) = ~7.4 GB — far exceeding the t3.large's available RAM. Two Qdrant features address this:

* **Scalar quantization (int8):** compresses each float32 (4 bytes) down to int8 (1 byte), reducing vector storage to ~1.85 GB for the full catalog. The accuracy tradeoff is negligible for ANN retrieval tasks. Enable this at collection creation time.
* **Memmap storage:** original (unquantized) vectors are stored on EBS disk rather than held in RAM. Qdrant only loads hot pages into memory on demand. This is the `on_disk` option in Qdrant's vector config.

With both enabled, Qdrant's working RAM footprint is approximately **2.5–3 GB** (quantized vectors + HNSW graph), leaving sufficient headroom for the other services.

**EC2 RAM budget (approximate):**

| Service | Est. RAM |
|---|---|
| Qdrant (quantized + memmap) | 2.5–3.0 GB |
| Postgres | 200–400 MB |
| Redis | 200–500 MB |
| API server | 300–500 MB |
| OS + Docker overhead | 500–800 MB |
| **Total** | **~4–5 GB** |

Create payload indexes for all fields you filter on — range indexes for numeric fields (`release_ts`, `runtime_minutes`, `maturity_rank`), keyword/array indexes for array fields (`genre_ids`, `watch_offer_keys`). This is required to keep selective filters fast and predictable.

---

### D) Redis (self-hosted Docker container)

Used for four caches:

1. **Embeddings cache** — key: `emb:{model}:{hash(normalized_text)}` (TTL ~days)
2. **Query understanding cache** — key: `qu:v{N}:{hash(normalized_query_text)}` (TTL 1 day)
3. **Trending set** — key: `trending:current`, a set of movie IDs (refreshed daily)
4. **TMDB detail cache** — key: `tmdb:movie:{movie_id}`, TMDB JSON blob (TTL 1 day)

---

### E) Nginx (self-hosted, runs directly on EC2)

* Reverse proxy routing HTTPS traffic to the API container
* HTTPS termination via Let's Encrypt / Certbot (free)
* Elastic IP attached to the EC2 provides a stable address for DNS

---

### F) Background jobs (cron on the EC2)

* **Daily** (midnight): trending refresh → Redis
* **Daily**: new movies → Postgres + Qdrant
* **Daily**: Postgres `pg_dump` + Qdrant snapshot → S3
* **Weekly**: watch offers refresh → Postgres + Qdrant

---

# 2) External APIs

### OpenAI

* Query understanding: ~24 parallel LLM calls (DAG) — skipped entirely on Redis cache hit
* Embeddings: original query + generated subqueries (same model everywhere) — Redis cache checked before each call

### TMDB

* Movie detail fetch on demand (only when user opens a detail page)
* Response cached in Redis for 1 day

---

# 3) Public API endpoints

### `POST /search`

**Request**

* `query: string`
* `filters?: { release, runtime, maturity, genres, watch_providers, watch_methods }`
* `shown_movie_counts?: [{ movie_id, count }]` (client-provided)

**Response (minimal)**

* `results: [{ movie_id, title, year, poster_url, score }]`
* Optional debug payload behind a feature flag (for dev)

> The search response includes only what's needed to render title cards. Detail page data comes separately via `GET /movie/{id}`.

---

### `GET /movie/{movie_id}`

* Check Redis for cached TMDB JSON → return immediately on hit
* On miss: call TMDB API → store in Redis (TTL 1 day) → return

**Why proxy TMDB through the server:** keeps API secrets off the client, enables caching, and gives one place to normalize/shape responses.

---

# 4) Data model (what lives where)

## 4.1 Postgres: minimal card + rerank metadata

A single "thin" table supporting card rendering and reranking metadata scoring:

* `movie_id (PK)`
* `tmdb_id`
* `title`
* `year`
* `poster_url`
* `release_ts` (BIGINT unix seconds)
* `runtime_minutes` (INT)
* `maturity_rank` (SMALLINT)
* `genre_ids` (INT[])
* `watch_offer_keys` (INT[])  ← encoded provider+method keys
* `audio_language_ids` (INT[])
* `reception_score` (FLOAT) (precomputed from imdb/metacritic)
* `updated_at` (optional)

Primary key is sufficient for enrichment — queries use `WHERE movie_id = ANY($1)` for bulk lookup.

## 4.2 Redis: four cache namespaces

| Key pattern | Contents | TTL |
|---|---|---|
| `emb:{model}:{hash}` | Embedding vector (float array) | ~7 days |
| `qu:v{N}:{hash}` | Full query understanding output blob (JSON) | 1 day |
| `trending:current` | Set of trending movie IDs | 1 day (refreshed daily) |
| `tmdb:movie:{id}` | TMDB detail JSON blob | 1 day |

The `qu:` key includes a prompt version prefix (`v{N}`). Bump the version whenever any system prompt changes — old keys expire harmlessly within a day without any explicit invalidation.

## 4.3 Qdrant: vectors + filter payload

**Collection**: `movies`

**Point structure**
* `id`: movie_id
* `vectors`: 8 named vectors (1536 dims each), stored with scalar quantization (int8) and memmap (`on_disk: true`)
* `payload`: `release_ts`, `runtime_minutes`, `maturity_rank`, `genre_ids`, `watch_offer_keys`

**Collection config (key settings)**
```
quantization_config:
  scalar:
    type: int8
    quantile: 0.99
    always_ram: false   # quantized vectors live in RAM; originals on disk

vectors_config:
  on_disk: true         # original float32 vectors stored on EBS, not RAM
```

## 4.4 S3: backups only

* Daily Postgres dump: `s3://your-bucket/backups/postgres/{YYYY-MM-DD}.dump`
* Daily Qdrant snapshot: `s3://your-bucket/backups/qdrant/{YYYY-MM-DD}.snapshot`

Cost is negligible (pennies per month at this data volume).

---

# 5) End-to-end request flow

## Step 0 — Request entry

* Generate `request_id`
* Create an in-memory request context with:
  * global deadline (target p95 3s)
  * stage timers + trace spans
  * in-flight futures for each DAG node

## Step 1 — Query Understanding DAG (LLMs, with Redis cache)

Before firing any LLM calls, check Redis:

* Normalize query text (lowercase, trim, collapse whitespace)
* Cache key: `qu:v{N}:{hash(normalized_query_text)}`
* **Cache hit** → deserialize the blob, skip the entire DAG, proceed to Step 2
* **Cache miss** → run the full DAG, serialize the complete output into Redis (TTL 1 day), then continue

The cached blob is the complete structured output of all DAG stages: extracted entities, metadata preferences, channel weights, and all per-vector subqueries. Always store it as one atomic key — never cache partial DAG outputs, as that creates consistency bugs between stages.

**DAG structure (on cache miss):**

Stage groups run in parallel; dependent stages start as soon as their parents resolve:
* Lexical entity extraction
* Vector subquery generation per channel
* Vector relevance per channel
* Channel weights
* Metadata preferences

**Engineering expectations for every LLM call:**
* Timeout set
* Bounded retries
* Structured logging (model, tokens, latency, error)
* If any required node fails after retries → fail the whole request

**Why this is safe to cache:** all DAG outputs are pure functions of the query text. `selected_filters`, `shown_movie_counts`, and trending state are applied downstream and are never baked into the cached result.

## Step 2 — Embedding (with Redis cache)

For each text to embed (original query + each generated subquery):

* Normalize string (lowercase, trim, collapse whitespace)
* Cache key: `emb:{model}:{hash(normalized_text)}`
* Cache hit → reuse stored vector
* Cache miss → call OpenAI embeddings API → store in Redis → continue

## Step 3 — Retrieval fan-out (runs concurrently)

### 3a) Lexical search (Postgres)

Triggered as soon as lexical entities are ready (may start before embedding completes).

* Query lexical tables / indexes
* Return `(movie_id, lexical_score, match_debug)` per candidate

### 3b) Vector search (Qdrant)

Triggered as soon as embeddings are available — anchor can start with just the original query embedding; other channels start as their subquery embeddings resolve.

For each channel search:
* Send query embedding + `limit = 2000`
* Apply hard filters via Qdrant payload filter
* Request minimal response: IDs + scores only (no payload)

**Channel search rules:**
* Always search Anchor with the original query embedding
* For channels with a generated subquery: search with the subquery embedding
* Also search every channel with relevance > not_relevant using the original query embedding

**The scores returned by Qdrant here are definitive.** There is no separate vector similarity recomputation at reranking time — no stored vectors are re-fetched after this step.

## Step 4 — Candidate merge + dedupe (in-memory)

Build a candidate map keyed by `movie_id`. For each candidate aggregate:

* `lexical_score` (if present; else 0)
* Per-channel vector scores from Qdrant (if a candidate appeared in both the subquery and original query results for a channel, keep the best score)
* Flags: "came from lexical / vector" + debug info

A candidate's vector score for a given channel is 0 if it did not appear in that channel's Qdrant results. No additional Qdrant fetches occur after this point.

## Step 5 — Metadata enrichment + trending set fetch (concurrent)

Run both concurrently:

**5a) Postgres bulk fetch**

Single query for all candidates:
```sql
SELECT movie_id, title, year, poster_url, release_ts, runtime_minutes,
       maturity_rank, genre_ids, watch_offer_keys, audio_language_ids, reception_score
FROM movie_card
WHERE movie_id = ANY($1)
```

**5b) Redis trending set fetch**

* Key: `trending:current`
* Load into an in-memory set for O(1) membership checks during reranking
* If the key is missing (e.g., daily job hasn't run), treat as empty and log a warning — do not fail the request

## Step 6 — Reranking (server-side)

Compute a final score per candidate from four components:

**1. Lexical score**
Use the score from Step 3a. If the candidate did not come from lexical search, this is 0.

**2. Vector score**
For each channel with relevance > not_relevant:
* Take the score returned directly from Qdrant in Step 3b (0 if not retrieved in that channel)
* Convert to a clamped z-score across all candidates for that channel
* Normalize to [0, 1]
* Multiply by that channel's normalized relevance weight

Sum across all active channels.

**3. Metadata preference score**
Score each extracted metadata preference against Postgres-enriched metadata. Applies to: `audio_language_ids`, `reception_score`, genre preferences, runtime, release year, maturity, and watch providers.

Each preference produces a score in [0, 1]. Average equally across all active preferences.

**Trending bonus:** if `prefers_trending_movies` is `True`, check the candidate's `movie_id` against the in-memory trending set from Redis. Present in set → bonus score contribution (scored as 1.0); absent → 0. This is averaged in as one metadata preference component alongside the others.

**4. Penalties**
* Exclude-entity overlap → heavy penalty
* Already shown in this session (from client-provided `shown_movie_counts`) → penalty that increases with shown count

**Final score formula:**
```
score = w_L * lexical_score + w_V * vector_score + w_M * metadata_score + P * session_penalty
```
where `P` is negative (higher shown count = lower score).

**Best practice:** log a full score breakdown for the top K results (e.g., top 20) behind a debug flag. This makes tuning and catching regressions significantly easier.

## Step 7 — Return top N

Sort by `final_score` descending. Return the minimal card payload:

* `movie_id, title, year, poster_url, final_score`

---

# 6) Movie detail retrieval (TMDB + Redis cache)

When user opens a detail page:

1. Client calls `GET /movie/{id}`
2. Server checks Redis (`tmdb:movie:{id}`):
   * Hit → return cached JSON immediately
   * Miss → call TMDB API → store in Redis (TTL 1 day) → return

---

# 7) Batch ingestion jobs

All jobs run as cron tasks on the EC2.

### Daily: new movies

1. Write new rows into Postgres `movie_card`
2. Generate 8 vectors per movie (offline, using the same embedding model)
3. Upsert into Qdrant (point id = movie_id, named vectors + payload filter fields)
4. Validate counts + run sampling checks

### Daily: trending refresh (midnight)

1. Fetch current trending movie IDs from upstream source
2. Overwrite `trending:current` in Redis (TTL 86400s)
3. Optionally persist to Postgres for auditing

### Daily: backups

1. `pg_dump` Postgres → upload to S3
2. Trigger Qdrant snapshot API → upload to S3

### Weekly: watch offers refresh

1. Recompute `watch_offer_keys` arrays
2. Update `movie_card.watch_offer_keys` in Postgres
3. Update Qdrant payload `watch_offer_keys` for affected points
4. If large update: batch with a clear version boundary to avoid half-old/half-new state mid-request

---

# 8) Deployment (single EC2 + Docker Compose)

### AWS resources

* **EC2 t3.large** (2 vCPU, 8 GB RAM) — runs all containers
* **EBS GP3, 250 GB** — attached storage for EBS/Postgres data and Qdrant memmap files
* **Elastic IP** — stable IP address for DNS (free when attached to a running instance)
* **S3 bucket** — daily backups (Postgres dump + Qdrant snapshot)
* **CloudWatch** — basic EC2 metrics (CPU, memory, disk) + billing alerts; free tier is sufficient

### Networking

* EC2 in a VPC with a public subnet (single-instance setup; no need for private subnet complexity)
* Security group: allow inbound 80 (HTTP), 443 (HTTPS), 22 (SSH from your IP only)
* All inter-service communication happens over Docker's internal network — Postgres, Redis, and Qdrant ports are NOT exposed to the internet

### Docker Compose services

```
services:
  api          # Python search API
  postgres     # Lexical DB + movie card metadata
  qdrant       # Vector DB
  redis        # All caches
  nginx        # Reverse proxy + HTTPS termination
```

Nginx listens on 443, terminates TLS (cert from Let's Encrypt / Certbot), and forwards to the API container on the internal Docker network.

### Estimated monthly cost

| Resource | Cost |
|---|---|
| EC2 t3.large (on-demand) | ~$60/mo |
| EC2 t3.large (1yr reserved) | ~$37/mo |
| EBS GP3 250 GB | ~$20/mo |
| S3 backups | ~$1–2/mo |
| OpenAI (LLM + embeddings) | ~$2–10/mo |
| Elastic IP | $0 (attached) |
| **Total (on-demand)** | **~$83–92/mo** |
| **Total (1yr reserved)** | **~$60–69/mo** |

---

# 9) Observability and debugging

Every `/search` request should emit:

* `request_id`
* Query understanding cache hit or miss
* Timings for:
  * LLM DAG total + per-node latency *(only on cache miss)*
  * Embedding calls (and per-call cache hit/miss)
  * Lexical query latency
  * Per-channel Qdrant query latency
  * Postgres metadata bulk fetch latency
  * Redis trending set fetch latency
  * Rerank latency
* Counts:
  * Candidates from lexical
  * Candidates from each vector channel
  * Merged candidates total
  * Final N returned

Use CloudWatch for EC2-level metrics (CPU, memory, disk). Structured JSON logs from the API container feed into CloudWatch Logs for per-request tracing. Set a billing alert at a comfortable threshold so unexpected cost spikes don't go unnoticed.

---

# 10) Practical engineering guidelines

* **Parallelize everything** after request entry. Never wait for all LLM output if a retrieval can start earlier.
* **Cache query understanding results.** On a hit, the entire DAG (the dominant latency source) is skipped. Bump `v{N}` whenever any system prompt changes.
* **Cache embeddings aggressively.** It's the cheapest latency and cost win available.
* **Use Qdrant scores directly.** The scores returned at retrieval time are authoritative — no stored vectors are re-fetched at reranking time.
* **Keep Qdrant responses lean** — IDs + scores only, no payload.
* **Bulk fetch metadata once per request.** Never query Postgres per-candidate.
* **Fetch the trending set from Redis once per request** and check membership in-memory — never query Redis per-candidate.
* **Keep reranking server-side** for consistency and to protect scoring logic.
* **Treat trending/watch offers as versioned inputs** so background job updates don't create half-old/half-new state mid-request.
* **Enable Qdrant scalar quantization and memmap at collection creation time** — these cannot be easily toggled after data is indexed without a full re-index.