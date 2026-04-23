# search_v2/ — V2 Search Pipeline

Four-stage query understanding and search execution pipeline. Replaces
the monolithic V1 pipeline with a structured decomposition that routes
queries to typed endpoint LLMs, executing retrieval against up to 7
specialized data sources.

## What This Module Does

Receives a user query (text), decomposes it through four sequential
stages, executes search across up to 7 endpoint types, and assembles
a ranked candidate list. Every stage is LLM-mediated except trending
(deterministic Redis lookup) and the final assembly/scoring (Stage 4).

## Pipeline Stages

```
Stage 1 (stage_1.py): Flow routing + creative alternatives
  → FlowRoutingResponse: primary_intent, alternative_intents, creative_alternatives
  → Determines SearchFlow (exact_title / similarity / standard / browse)

Stage 2A (stage_2a.py): Concept extraction
  → Step2AResponse: list of concept strings + per-phrase unit_analysis traces
  → Decompose-first flow: per-phrase verdicts before grouping

Stage 2B (stage_2b.py): Expression planning (one call per concept)
  → QueryConcept: list of RetrievalExpression (dealbreaker/preference, EndpointRoute)
  → Step 2B failures drop only that concept; Step 2A failure raises

Stage 3 (stage_3/): Endpoint translation + execution (7 endpoints)
  ├── entity_query_generation.py + entity_query_execution.py
  ├── metadata_query_generation.py + metadata_query_execution.py
  ├── award_query_generation.py + award_query_execution.py
  ├── franchise_query_generation.py + franchise_query_execution.py
  ├── keyword_query_generation.py + keyword_query_execution.py
  ├── semantic_query_execution.py
  └── trending_query_execution.py
  All executors return EndpointResult (list[ScoredCandidate]).

Stage 4 (stage_4/): Assembly & reranking
  → Flattens expressions → concept-level inclusion/exclusion aggregation
  → run_stage_4() is the public entry point
```

## Key Files

| File | Purpose |
|------|---------|
| `stage_1.py` | Flow routing LLM call. `route_query()` returns `FlowRoutingResponse`. System prompt (~21KB) has 4 modular sections. `creative_alternatives` field generates productive sub-angles on broad queries (separate from `alternative_intents` which are competing readings). Prompt enforces: evaluative-word preservation in `intent_rewrite` (best/top/great), brevity on `creative_spin_analysis` parentheticals (~8 words), readings-enumeration trace for vagueness-vs-ambiguity distinction. |
| `stage_2a.py` | Concept extraction LLM call. `run_stage_2a()` returns `Step2AResponse`. Prompt uses decompose-first flow (per-phrase verdicts in pass 1, fuse/split in pass 2), `interpret` verdict (replaces `best_guess`), endpoint descriptions in user-facing capability language, fusion criterion (same-family AND same sub-dimension AND ranking-style). |
| `stage_2b.py` | Expression planning per concept. Called once per concept from `run_stage_2()`. Returns `QueryConcept` with typed `RetrievalExpression` items. |
| `stage_2.py` | Orchestrator. `run_stage_2()`: single Step 2A call then parallel Step 2B per concept. Step 2B failures drop the concept; all Step 2B dropped → raises (guards against silent browse fallback). |
| `stage_3/` | One generator + one executor module per endpoint. Generators call the shared LLM router and return `(output, input_tokens, output_tokens)`. Executors take the output spec and `restrict_to_movie_ids: set[int] | None` (None = dealbreaker/candidate-generating; set = preference/score-only; empty set = short-circuit) and return `EndpointResult`. |
| `stage_4/__init__.py` | `run_stage_4()` public entry point. `__getattr__` lazy-imports keep package imports lightweight. |
| `stage_4/orchestrator.py` | Flattens expressions into tagged runtime items preserving concept identity. Coordinates dealbreaker execution, preference execution, and score assembly. |
| `stage_4/scoring.py` | Concept-level inclusion/exclusion aggregation: sums per-concept maxima for inclusion dealbreakers, applies one max-based penalty per semantic-exclusion concept. Preferences are independent even within the same concept. |
| `stage_4/flow_detection.py` | Detects browse flow: triggers when no non-semantic inclusion dealbreakers exist. Browse seeding: `popularity_score DESC NULLS LAST, movie_id DESC`. |
| `stage_4/types.py` | Runtime types internal to Stage 4. |
| `stage_4/priors.py` | Deleted — quality/notability priors removed from V2 runtime. |

## Endpoint Contract

All 7 executors follow the same dual-mode signature:
- `restrict_to_movie_ids=None` → dealbreaker mode: executor generates
  its own candidate pool and returns scored results
- `restrict_to_movie_ids=set[int]` → preference mode: executor scores
  exactly the supplied ids (0.0 for absent movies)
- `restrict_to_movie_ids=set()` → short-circuit: returns empty
  `EndpointResult` immediately

Failures retry once; second failure returns empty `EndpointResult`
(soft failure — never raises to the orchestrator).

## Key Patterns

- **Positive-presence invariant**: all step-3 endpoint specs express
  presence of an attribute, not absence. Exclusion direction is carried
  by the `DealbreakDirection` field on the expression.
- **Concept-level aggregation in Stage 4**: expressions from the same
  concept contribute to one inclusion score (max) and one exclusion score
  (max penalty). This prevents a concept with multiple expressions from
  dominating the score unfairly.
- **Token-index resolution**: franchise and award endpoints resolve LLM
  surface forms via `postgres.py` posting-list helpers
  (`fetch_franchise_entry_ids_for_tokens`,
  `fetch_award_name_entry_ids_for_tokens`). Empty token resolution →
  empty `EndpointResult`, not a silently broadened query.
- **Step 2A `interpret` verdict**: replaces `best_guess`. Emits 1+
  retrievable atoms each tagged with exactly one retrieval family
  (literal/metadata/keyword/semantic). Decompose-first pass prevents
  compound-string collapsing.

## Interactions

- `schemas/` — all step-3 translation schemas, `EndpointResult`,
  `ScoredCandidate`, flow-routing and query-understanding schemas.
- `db/postgres.py` — posting-list reads for franchise, award, entity,
  studio endpoints.
- `db/qdrant.py` — semantic endpoint vector searches.
- `db/redis.py` — trending endpoint hash reads.
- `implementation/llms/generic_methods.py` — shared LLM router used by
  all generator modules.
- `implementation/misc/franchise_text.py` — `tokenize_franchise_string`
  (FRANCHISE_STOPLIST applied here, not in executor).
- `implementation/misc/award_name_text.py` — `tokenize_award_string_for_query`
  (AWARD_QUERY_STOPLIST applied here; differs from ingest tokenizer which
  writes every token).

## Gotchas

- **Stoplist asymmetry for awards**: ingest writes every token to
  `lex.award_name_token`; query drops `AWARD_QUERY_STOPLIST` before
  posting-list fetch. The asymmetry is intentional — lets the droplist
  be revised without re-ingesting.
- **Franchise backfill precondition**: `lex.franchise_token` must be
  rebuilt under the stopword-dropping `tokenize_franchise_string` before
  the franchise executor ships. Stale tokens (`the`, `of`, `and`, etc.)
  in the index inflate match sets.
- **`FranchiseQuerySpec.franchise_or_universe_names`** (renamed from
  `lineage_or_universe_names`). Any planning-doc references to the old
  name are superseded.
- **Razzie dual-axis opt-in**: the default `ceremony_id <> 10` Razzie
  exclusion is overridden when `category_tags` contains any id in
  `RAZZIE_TAG_IDS`. Both axes signal intent independently — either alone
  suffices.
- **`stage_4/priors.py` is deleted.** Any code importing it will fail;
  no replacement — priors are out of scope for V2 runtime.
- **Stage 2 `run_stage_2()` raises on total Step 2B loss.** If Step 2A
  found concepts but every Step 2B call failed or was dropped, the
  function raises rather than returning an empty concept list. This
  prevents silent browse degradation.
- **Semantic dealbreaker scoring uses a top-2000 probe** that doubles
  as both the elbow/floor calibration distribution and the candidate
  pool (single Qdrant call). `kneed>=0.8` is a required dependency for
  Kneedle elbow detection.
