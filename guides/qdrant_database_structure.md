# Qdrant Database Structure Guide

## 1) What Qdrant is responsible for in this project

Qdrant is **only** the **dense vector retrieval engine** + **hard-filter executor**.

It stores:

* **8 dense embeddings per movie** (named vectors), each representing a different “lens” on the movie.
* A **minimal payload** used for **fast hard filters** (year/runtimes/maturity/genres/watch providers).

It does **not** store:

* Full movie cards (title, poster URL, overview, cast, etc.). Those live in Postgres / TMDB and are fetched during enrichment.

Why this split:

* Qdrant stays lean and fast (vector IO + filters).
* Postgres stays the source of truth for structured metadata and UI hydration.

---

## 2) Collections and naming

### 2.1 Single logical collection

We use **one logical collection** for all movies:

* Logical name (what the app uses): `movies`
* Physical name (versioned): `movies_v{N}` (e.g., `movies_v1`, `movies_v2`)

### 2.2 Alias strategy (zero-downtime rebuilds)

Use a Qdrant **alias** so you can rebuild embeddings without downtime:

* Build new physical collection: `movies_v2`
* Bulk ingest into `movies_v2`
* Atomically move alias `movies` from `movies_v1` → `movies_v2`

Alias operations are designed to be atomic at the API level. ([python-client.qdrant.tech][1])

**Decision:** Use aliasing from day 1. It’s cheap and prevents “migration debt.”

---

## 3) Point identity (IDs)

### 3.1 Point ID type

Each point is one movie:

* `point_id = movie_id`
* **Type:** **unsigned 64-bit integer** (`uint64`)
  (Qdrant point IDs can be numeric or UUID; numeric IDs are a great fit here.)

**Decision:** Use numeric `uint64` IDs everywhere (Qdrant + Postgres).
Avoid UUIDs unless you truly need cross-system uniqueness without coordination.

---

## 4) Vectors schema (named vectors)

Each movie stores **8 named dense vectors** in the same point.

### 4.1 Vector names

These names should exactly match your project vocabulary and query-understanding routing:

* `anchor`
* `plot_events`
* `plot_analysis`
* `viewer_experience`
* `watch_context`
* `narrative_techniques`
* `production`
* `reception`

### 4.2 Vector dimensionality + distance

* **Dimensions:** `1536` each
* **Distance metric:** `Cosine`

Qdrant implements cosine by normalizing vectors at insert-time, then using dot-product at query-time for speed. ([Qdrant][2])

**Decision:** Use `Cosine` for all 8 vectors.

---

## 5) Payload schema (minimal hard-filter fields)

We intentionally keep payload small. These fields exist purely to support fast filtering and stable retrieval latency.

### 5.1 Payload fields and types

All numeric payloads should be treated as **int64-compatible** in your pipeline (even if they fit in smaller ranges).

```json
{
  "release_ts": 946684800,
  "runtime_minutes": 123,
  "maturity_rank": 2,
  "genre_ids": [18, 53],
  "watch_offer_keys": [800102, 8_00103]
}
```

#### Field definitions

**`release_ts: int`**

* Unix timestamp (seconds) for release date.
* Used for year/time-window filters (range queries).

**`runtime_minutes: int`**

* Movie runtime minutes.
* Used for range filters.

**`maturity_rank: int`**

* A **monotonic integer** where “more permissive” is larger.
* Enables filters like “<= PG-13” as a numeric range.

Recommended mapping (example):

* `0=G, 1=PG, 2=PG-13, 3=R, 4=NC-17, 5=Unrated/Unknown`

**Decision:** Use this mapping unless you have a better canonical source.

**`genre_ids: int[]`**

* TMDB genre IDs (or your own canonical genre IDs).
* Used for “must include any of these genres” filters.

**`watch_offer_keys: int[]`**

* Encodes (provider, method) into a single integer for compact matching.

**Final encoding decision (simple, reversible, future-proof):**

* `watch_offer_key = provider_id * 100 + method_code`
* where `method_code` is:

  * `1=stream`
  * `2=rent`
  * `3=buy`

Example:

* Netflix streaming if `provider_id=8` → `8*100 + 1 = 801`

This prevents collisions and supports expansion (more methods) without schema changes.

---

## 6) Payload indexing strategy

Hard filters must remain fast at scale, so we index all filterable payload fields.

Qdrant supports an `integer` payload index with two relevant capabilities:

* `lookup` (match / direct lookup)
* `range` (range filters)

The docs explicitly describe configuring these flags and warn about enabling both in certain cases. ([Qdrant][3])

### 6.1 Index plan

| Field              | Type  | Query style | Index config                                                  |
| ------------------ | ----- | ----------- | ------------------------------------------------------------- |
| `release_ts`       | int   | range       | integer index (`range=true`, `lookup=false`) + mark principal |
| `runtime_minutes`  | int   | range       | integer index (`range=true`, `lookup=false`)                  |
| `maturity_rank`    | int   | range       | integer index (`range=true`, `lookup=false`)                  |
| `genre_ids`        | int[] | match any   | integer index (`lookup=true`, `range=false`)                  |
| `watch_offer_keys` | int[] | match any   | integer index (`lookup=true`, `range=false`)                  |

### 6.2 Principal optimization for `release_ts`

Qdrant supports “principal” optimization for certain index types, including integer. (This is meant for commonly-used filters like timestamps.) ([Qdrant][3])

**Decision:** Set `release_ts` as the principal indexed field because year filtering is common in movie search.

---

## 7) Collection configuration (performance + memory)

This project’s key constraint is memory: 150k movies × 8 vectors × 1536 dims × float32 is large.

### 7.1 Storage decisions

**Decision A — Store original float vectors on disk (memmap):**

* Set `on_disk: true` for each vector config.

**Decision B — Enable scalar quantization (int8) and keep quantized vectors in RAM:**

* Scalar quantization config:

  * `type: int8`
  * `quantile: 0.99`
  * `always_ram: true`

This is the corrected and recommended setup for “originals on disk, quantized in RAM.” The Qdrant quantization guide shows `always_ram: true` and explains the feature. ([Qdrant][4])

### 7.2 Rescoring (accuracy recovery)

At query time:

* Use quantized vectors for fast candidate generation.
* Then **rescore** top results with original vectors (on disk).

Qdrant recommends disabling rescore only when originals are on slow storage (HDD/network). With SSD/EBS, rescore is typically fine and improves quality. ([Qdrant][4])

**Decision:** Keep `rescore=true` by default.

### 7.3 Oversampling (quality knob)

Oversampling means: fetch extra candidates from the quantized stage, then rescore and return the best `limit`. Qdrant defines oversampling explicitly. ([Qdrant][4])

**Decision:** Start with:

* `oversampling = 2.0` for high-quality recall
* tune down later if latency is too high

### 7.4 HNSW defaults + tuning

HNSW key knobs are `m`, `ef_construct`, and query-time `hnsw_ef`. Qdrant’s guidance: higher values improve recall but cost memory/time. ([Qdrant][5])

**Decision (pragmatic starting point):**

* Use Qdrant defaults initially, but set **query-time**:

  * `hnsw_ef = 128`
* Only tune `m` / `ef_construct` after you have recall/latency measurements.

---

## 8) Concrete “create collection” schema

Below is the intended structure (conceptual YAML). Your actual code will use REST or the Python client.

```yaml
collection: movies_v1
alias: movies

vectors:
  anchor:              { size: 1536, distance: Cosine, on_disk: true }
  plot_events:         { size: 1536, distance: Cosine, on_disk: true }
  plot_analysis:       { size: 1536, distance: Cosine, on_disk: true }
  viewer_experience:   { size: 1536, distance: Cosine, on_disk: true }
  watch_context:       { size: 1536, distance: Cosine, on_disk: true }
  narrative_techniques:{ size: 1536, distance: Cosine, on_disk: true }
  production:          { size: 1536, distance: Cosine, on_disk: true }
  reception:           { size: 1536, distance: Cosine, on_disk: true }

quantization_config:
  scalar:
    type: int8
    quantile: 0.99
    always_ram: true

payload_indexes:
  release_ts:       { type: integer, lookup: false, range: true, is_principal: true }
  runtime_minutes:  { type: integer, lookup: false, range: true }
  maturity_rank:    { type: integer, lookup: false, range: true }
  genre_ids:        { type: integer, lookup: true,  range: false }
  watch_offer_keys: { type: integer, lookup: true,  range: false }
```

---

## 9) Query patterns used by the server

### 9.1 Per-vector retrieval behavior

From query understanding, each vector channel is either:

* **relevant** → run Qdrant search using that vector name
* **not_relevant** → skip that channel entirely

**Decision:** Skip irrelevant vectors to reduce fanout latency and cost.

### 9.2 Result size (candidate pool)

Per relevant vector, retrieve up to:

* `limit = 2000`
* return only: `(id, score)` (no payload, no vectors)

**Decision:** Always query with:

* `with_payload = false`
* `with_vector = false`

### 9.3 Use Query API (preferred)

Qdrant’s Query API is the unified interface for search + filtering + advanced modes. ([Qdrant][6])

### 9.4 Batch queries (reduces network overhead)

Instead of 8 separate HTTP requests, you can send multiple queries in **one request** using the batch Query endpoint. ([api.qdrant.tech][7])

**Decision:** Use **batch query** once you want to shave p95 latency and reduce request overhead.

---

## 10) Ingestion and update flows

### 10.1 Initial bulk ingest

For each movie:

1. Compute/store all 8 embeddings
2. Build payload (filters only)
3. Upsert into Qdrant in large batches (record-oriented)

Qdrant supports batch uploads for points to reduce network overhead. ([Qdrant][8])

### 10.2 Weekly watch-offer refresh

Watch providers change; embeddings usually do not.

**Decision:** Update only the payload field:

* recompute `watch_offer_keys`
* call payload update for affected `movie_id`s (don’t rewrite vectors)

Operationally, you want to avoid a “half old / half new” world mid-request:

* batch updates and use a clear “version boundary” in your job design (application-level consistency).

---

## 11) Backups and monitoring (required ops minimum)

### 11.1 Backups (snapshots)

**Decision:** Take a **daily Qdrant snapshot** and ship it to S3 (alongside Postgres dumps).
This gives you disaster recovery without rebuilding embeddings.

### 11.2 Monitoring

At minimum, wire:

* `/metrics` and `/telemetry` for Qdrant health and sizing
* health endpoints: `/healthz`, `/livez`, `/readyz`

Qdrant documents these endpoints and how metrics/telemetry are exposed. ([Qdrant][9])

---

## 12) What this data represents conceptually

Each vector is a different semantic “view” of a movie. At query time:

* the system generates per-vector subqueries (or uses the user query directly),
* embeds them,
* retrieves candidate movie IDs per vector,
* fuses + reranks candidates downstream (outside Qdrant).

Qdrant’s job is to give you a **high-recall candidate set** under **strict hard filters** with predictable latency.