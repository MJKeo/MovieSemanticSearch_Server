"""Telemetry name registry — the single source of truth for span names and
attribute keys emitted by our *manual* OpenTelemetry instrumentation.

Every manual span name and attribute key is defined here exactly once, as a
`Name` derived from its namespace root, so the root token (`movie_credits`, …)
is never retyped and a typo can't silently split a metric across two spellings.
Import the constants at call sites; never write a raw name string inline.

This module is intentionally dependency-free (stdlib only — no `opentelemetry`
import), so a search/domain module can reference a name without pulling in the
SDK.

Scope: this registry owns the *names* of manual spans and attributes only. It
does NOT own:
  - Attribute *values* — a closed value set is a `str`-Enum in its owning module
    (`MoviePayloadSource` in `api/main.py`, `FailureReason` in `api/outcome.py`),
    not a Name.
  - Span *event* messages ("cache read failed") — human-readable messages, not
    queryable keys.
  - Standard OTel keys (`http.*`, `db.*`, `net.*`, `gen_ai.*`, …) — those come
    from auto-instrumentation or the semantic conventions and are never authored
    or re-spelled here.


NAMING RULES
============

A. Structure. Every name is `namespace.leaf` — at least one namespace, never a
   bare key (attribute keys share one flat, global keyspace across every span,
   metric, and log; a bare `source`/`query` collides). Names are defined once as
   a `Name` via `.child()`; call sites reference the constant. Two levels is the
   default depth.

B. Root. The root is the domain/endpoint that conceptually OWNS the thing
   (`movie`, `movie_details`, `movie_credits`, `cache`, `title_search`). For a
   span, the owner is its home endpoint even when it runs under another —
   `movie_credits.build_and_cache` keeps that name under a /movie_details
   request, because runtime parent/child is the trace tree, not the name. Never
   author or reuse a standard OTel root.

C. Dot vs. underscore (the one decision, applied at every boundary). When you
   add a segment, choose a dot (new namespace level) only when it:
     1. already groups >= 2 emitted attributes, OR
     2. has >= 2 concrete siblings you will ACTUALLY emit on the foreseeable
        roadmap (name them — if you can't, it doesn't qualify).
   Otherwise fold it into the leaf with an underscore. "Expansion" means
   telemetry we'll emit, not properties the real-world object happens to have
   (every noun has properties; that's not the test). When genuinely 50/50, stay
   flat — promoting flat -> namespace later is a one-line edit here.

     cache.write_ok          -> dot: `cache` is a real, growing telemetry space
                                (hit, read_ok, ttl_seconds are nameable siblings)
     outcome.failure_reason  -> dot: `outcome` already groups >= 2 emitted
                                attributes (success + failure_reason)
     movie.payload_source    -> underscore: we measure one fact (provenance);
                                we will never emit movie.payload.size/.bytes
     gen_ai.usage.input_tokens (future) -> dot: `usage` groups input/output/total
                                (illustrative — a standard key, not ours to mint)

D. Leaf. snake_case; name the what/why. Counts end `_count`; booleans read as a
   predicate (`_ok`); measures carry a unit suffix (`_seconds`, `_ms`, `_chars`).
   The key is fixed vocabulary — never bake a varying/high-cardinality value into
   it.

E. Values are not names. A closed set of possible values is a `str`-Enum in the
   owning module, set via `.value`. Names (this module) and values (enums) are
   deliberately different kinds of thing.

F. Cardinality. A namespaced span attribute may be high-cardinality
   (`movie.tmdb_id`, `title_search.query`) — fine on spans. That same key must
   NEVER become a metric label (Phase 3): only low-cardinality keys
   (`movie.payload_source`, `outcome.success`, `outcome.failure_reason`) are
   label-eligible.
"""

from __future__ import annotations


class Name(str):
    """A telemetry name that is also a namespace.

    It *is* its fully-qualified dotted string (subclasses `str`), so it hands
    straight to `start_as_current_span(...)` / `set_attribute(...)` with no
    unwrapping. `.child()` derives a deeper name, so any token — root or leaf —
    can grow subtokens later without ever being migrated out of a raw string.

    `.child()` takes a bare segment string, never another `Name`. That keeps
    name nesting (a declaration-time parent/child of *strings*) distinct from
    span nesting (a runtime parent/child of *spans* via trace context): a span's
    name never absorbs its runtime parent's name.
    """

    def child(self, segment: str) -> "Name":
        """Derive `self.<segment>` as a new Name (dot-joined)."""
        return Name(f"{self}.{segment}")


# --- outcome: per-request success verdict, on every endpoint's server span ---
# `outcome` earns a dot (rule C) because it groups >= 2 emitted attributes:
# `success` is always set, `failure_reason` only when success is false. Both are
# low-cardinality, so both stay metric-label-eligible (rule F).
OUTCOME = Name("outcome")
OUTCOME_SUCCESS = OUTCOME.child("success")                # outcome.success (bool)
OUTCOME_FAILURE_REASON = OUTCOME.child("failure_reason")  # outcome.failure_reason

# --- movie: cross-endpoint movie attributes (shared by both TMDB endpoints) ---
MOVIE = Name("movie")
MOVIE_TMDB_ID = MOVIE.child("tmdb_id")                    # movie.tmdb_id
MOVIE_PAYLOAD_SOURCE = MOVIE.child("payload_source")      # movie.payload_source

# --- movie_details: pipeline spans ---
MOVIE_DETAILS = Name("movie_details")
MOVIE_DETAILS_PAYLOAD_CREATION = MOVIE_DETAILS.child("payload_creation")  # span
MOVIE_DETAILS_CACHE_WRITE = MOVIE_DETAILS.child("cache_write")            # span

# --- movie_credits: pipeline spans + build attributes ---
MOVIE_CREDITS = Name("movie_credits")
MOVIE_CREDITS_PAYLOAD_CREATION = MOVIE_CREDITS.child("payload_creation")  # span
MOVIE_CREDITS_BUILD_AND_CACHE = MOVIE_CREDITS.child("build_and_cache")    # span
MOVIE_CREDITS_CAST_COUNT = MOVIE_CREDITS.child("cast_count")              # attr
MOVIE_CREDITS_CREW_COUNT = MOVIE_CREDITS.child("crew_count")              # attr

# --- cache: shared best-effort cache-write outcome ---
CACHE = Name("cache")
CACHE_WRITE_OK = CACHE.child("write_ok")                  # cache.write_ok

# --- title_search: typeahead request attributes ---
TITLE_SEARCH = Name("title_search")
TITLE_SEARCH_QUERY = TITLE_SEARCH.child("query")
TITLE_SEARCH_LIMIT = TITLE_SEARCH.child("limit")
TITLE_SEARCH_RESULT_COUNT = TITLE_SEARCH.child("result_count")
TITLE_SEARCH_FUZZY_RESULT_COUNT = TITLE_SEARCH.child("fuzzy_result_count")

# --- query_search: request-boundary input attributes ---
# Written at handler entry from the RAW wire body, before validation runs, so a
# rejected (400/422) trace still carries the input that caused it. The text
# attrs are defensively truncated at the call site (Pydantic enforces no max
# length on these fields); the `_chars` attrs carry the true pre-truncation
# length (rule D unit suffix). All four are high-cardinality — span-attr-only,
# never metric labels (rule F).
QUERY_SEARCH = Name("query_search")
QUERY_SEARCH_QUERY = QUERY_SEARCH.child("query")
QUERY_SEARCH_QUERY_CHARS = QUERY_SEARCH.child("query_chars")
QUERY_SEARCH_CLARIFICATION = QUERY_SEARCH.child("clarification")
QUERY_SEARCH_CLARIFICATION_CHARS = QUERY_SEARCH.child("clarification_chars")
# Request-level cost rollup, written on the server span at stream end: the sum
# of every LLM + embedding call's USD cost incurred by the request (all billed
# attempts, including retried/failed ones — see observability/cost_tracking.py).
# Stays flat (rule C): there is no second `cost.*` sibling on the roadmap, so a
# leaf with the established `_usd` unit suffix (mirroring `llm.cost_usd`) is
# correct. A continuous measure — span-attr-only, never a metric label (rule F).
QUERY_SEARCH_COST_USD = QUERY_SEARCH.child("cost_usd")
# Request-level token rollup, written on the server span at stream end beside
# `cost_usd`: the summed token usage across every LLM + embedding call (all
# billed attempts, mirroring cost). `usage` earns a dot (rule C): it groups
# three emitted siblings and mirrors the standard `gen_ai.usage.*` triad. Each
# is a continuous count — span-attr-only, never a metric label (rule F).
# `cached_input_tokens` is a SUBSET of `input_tokens` (cached ⊆ input), never
# additive to it.
QUERY_SEARCH_USAGE = QUERY_SEARCH.child("usage")
QUERY_SEARCH_USAGE_INPUT_TOKENS = QUERY_SEARCH_USAGE.child("input_tokens")
QUERY_SEARCH_USAGE_CACHED_INPUT_TOKENS = QUERY_SEARCH_USAGE.child("cached_input_tokens")
QUERY_SEARCH_USAGE_OUTPUT_TOKENS = QUERY_SEARCH_USAGE.child("output_tokens")

# --- query_search: Step 0 / Step 1 pipeline spans (1c-1 Bite 3) ---
# Step 0 (flow routing) and Step 1 (spin generation) run as a parallel LLM
# pair at the head of the pipeline. Each gets a manual span so the router's
# `llm.generate` child nests beneath the right step (step identity comes from
# this nesting, never duplicated onto the LLM span). These spans are owned by
# `query_search` (rule B / OQ #1): /rerun_query_search reuses the Step 2 ->
# Stage 4 spans, not routing, so query_search is routing's home endpoint.
# Attributes stay FLAT under the query_search root (underscore leaves), matching
# the existing query_search.* input attrs — step_0 emits two facts but the local
# convention keeps siblings flat (cf. query / query_chars), and rule C says stay
# flat when in doubt (flat -> namespace is a one-line edit later).
#   step_0_flows:                 activated flow names (SearchFlow values +
#                                 "standard" when it co-fires). Low-cardinality
#                                 (closed set), array-valued; never empty on
#                                 success. Metric-label-ineligible only because
#                                 it is an array (rule F is about cardinality).
#   step_0_standard_branch_count: the standard-flow branch budget Step 0 sets
#                                 (0 when standard doesn't fire). Always present
#                                 on success. Low-cardinality, label-eligible.
#   step_1_unused:                true when routing left no budget for spins, so
#                                 Step 1's output feeds no branch. Derivable from
#                                 the two step_0 attrs above, but recorded
#                                 directly so the verdict is legible on the span
#                                 itself rather than reconstructed across sources.
QUERY_SEARCH_STEP_0 = QUERY_SEARCH.child("step_0")                    # span name
QUERY_SEARCH_STEP_0_FLOWS = QUERY_SEARCH.child("step_0_flows")        # attr (str[])
QUERY_SEARCH_STEP_0_STANDARD_BRANCH_COUNT = QUERY_SEARCH.child(
    "step_0_standard_branch_count"
)                                                                    # attr (int)
QUERY_SEARCH_STEP_1 = QUERY_SEARCH.child("step_1")                    # span name
QUERY_SEARCH_STEP_1_UNUSED = QUERY_SEARCH.child("step_1_unused")      # attr (bool)

# --- query_search: Step 2 / Step 3 / query-generation spans (1c-1 Bite 4) ---
# The per-branch trait pipeline: Step 2 (trait extraction, one LLM call per
# standard branch) -> per-trait fan-out of Step 3 (trait decomposition, one LLM
# call per trait) -> query generation (one handler-LLM call per surviving
# category call). Each stage's `llm.generate` child nests under the manual span
# here; the step-level output is captured as low-cardinality summary attributes
# so it stays queryable when LLM payload sampling is dialed down. All spans are
# owned by `query_search` and sit under the branch span (rule B / OQ #1),
# alongside step_0/1/branch. Attributes stay FLAT under the root (underscore
# leaves), matching the step_0_* / step_1_* / branch_* precedent (rule C).
#
# step_2 (one per standard branch, sibling of the trait spans — it closes at the
# Step-2 LLM return, the trait spans start after):
#   step_2_trait_count:            number of traits Step 2 committed (int).
#   step_2_contextualized_phrases: the traits' contextualized_phrase strings, in
#                                  order — the at-a-glance "what did Step 2
#                                  decide" list. High-cardinality span attr,
#                                  never a metric label (rule F).
QUERY_SEARCH_STEP_2 = QUERY_SEARCH.child("step_2")                    # span name
QUERY_SEARCH_STEP_2_TRAIT_COUNT = QUERY_SEARCH.child(
    "step_2_trait_count"
)                                                                    # attr (int)
QUERY_SEARCH_STEP_2_CONTEXTUALIZED_PHRASES = QUERY_SEARCH.child(
    "step_2_contextualized_phrases"
)                                                                    # attr (str[])

# trait (one per Step-2 trait, brackets that trait's Step 3 + query generation).
# Its attributes identify the trait; combine_mode / categories live on the nested
# step_3 span. `trait_step_3_error` is set only on a Step-3 soft-fail (retries
# exhausted) — a degradation: the trait span stays UNSET and the request verdict
# is untouched; the nested `llm.generate` child carries the ERROR status.
#   trait_phrase:      the trait's contextualized_phrase (str; high-cardinality).
#   trait_polarity:    Polarity value — positive / negative (closed set).
#   trait_commitment:  required / elevated / neutral / supporting / diminished
#                      (closed set). The `"solo trim"` span EVENT message (kept
#                      category + dropped count) is a call-site string, not a Name.
QUERY_SEARCH_TRAIT = QUERY_SEARCH.child("trait")                      # span name
QUERY_SEARCH_TRAIT_PHRASE = QUERY_SEARCH.child("trait_phrase")        # attr (str)
QUERY_SEARCH_TRAIT_POLARITY = QUERY_SEARCH.child("trait_polarity")    # attr (str)
QUERY_SEARCH_TRAIT_COMMITMENT = QUERY_SEARCH.child(
    "trait_commitment"
)                                                                    # attr (str)
QUERY_SEARCH_TRAIT_STEP_3_ERROR = QUERY_SEARCH.child(
    "trait_step_3_error"
)                                                                    # attr (str)

# step_3 (one per trait, wraps the Step-3 LLM call). Records the POST-trim state:
# combine_mode and the categories that actually survive the SOLO trim, not the
# raw committed list.
#   step_3_combine_mode: TraitCombineMode value — solo / framings / facets.
#   step_3_categories:   category names of the surviving category calls (str[]).
QUERY_SEARCH_STEP_3 = QUERY_SEARCH.child("step_3")                    # span name
QUERY_SEARCH_STEP_3_COMBINE_MODE = QUERY_SEARCH.child(
    "step_3_combine_mode"
)                                                                    # attr (str)
QUERY_SEARCH_STEP_3_CATEGORIES = QUERY_SEARCH.child(
    "step_3_categories"
)                                                                    # attr (str[])

# query_generation (one per category call that runs a handler LLM — the
# EXPLICIT_NO_OP and NO_LLM_PURE_CODE buckets return before the span, so
# deterministic/no-op calls get no span).
#   query_generation_category:  the category name this handler call routes (str).
#   query_generation_endpoints: the EndpointRoute names that actually fired for
#                               this call (str[]) — just which endpoints
#                               activated; the detailed params live on the
#                               nested `llm.generate` payload. Empty when the
#                               handler fired nothing.
QUERY_SEARCH_QUERY_GENERATION = QUERY_SEARCH.child(
    "query_generation"
)                                                                    # span name
QUERY_SEARCH_QUERY_GENERATION_CATEGORY = QUERY_SEARCH.child(
    "query_generation_category"
)                                                                    # attr (str)
QUERY_SEARCH_QUERY_GENERATION_ENDPOINTS = QUERY_SEARCH.child(
    "query_generation_endpoints"
)                                                                    # attr (str[])

# --- query_search: per-branch span (1c-1 Bite 4) ---
# The pipeline fans out into concurrent branches — up to three standard branches
# (main query + spins) plus at most one entity flow (exact_title, similarity,
# franchise, studio, person). Each branch gets a manual span bracketing its full
# lifecycle (launch -> terminal branch_results) so its `llm.generate` children
# (Step 2 / Step 3 / Stage 4) nest beneath the right branch instead of directly
# under the request span. Owned by `query_search` (rule B), same as the Step 0/1
# spans. Attributes stay FLAT under the query_search root (underscore leaves),
# matching the step_0_* / step_1_* precedent (rule C: stay flat when the local
# sibling pattern is flat).
#   branch_type:              the fetch type — one of standard / exact_title /
#                             similarity / non_character_franchise /
#                             character_franchise / studio / person. Closed,
#                             low-cardinality set (mirrors SearchFlow values);
#                             set on every branch span.
#   branch_uses_original_text: true only for the first standard branch of the
#                             non-clarification flow (the one searching the typed
#                             query verbatim rather than a generated/rewritten
#                             query). Set on standard branches only; false for
#                             spins and for every branch on the rerun path.
QUERY_SEARCH_BRANCH = QUERY_SEARCH.child("branch")                    # span name
QUERY_SEARCH_BRANCH_TYPE = QUERY_SEARCH.child("branch_type")          # attr (str)
QUERY_SEARCH_BRANCH_USES_ORIGINAL_TEXT = QUERY_SEARCH.child(
    "branch_uses_original_text"
)                                                                    # attr (bool)

# --- query_search: entity-flow (non-standard branch) attributes (1c-1 Bite 8) ---
# The six entity flows (exact_title / similarity / non_character_franchise /
# character_franchise / studio / person) are deterministic resolve->hydrate
# branches: Step 0 already asserted the entity exists, so an empty result means
# something concrete broke (name didn't resolve, resolved to the wrong entity,
# filters emptied the pool, or a genuine catalog gap). These attributes make
# that resolution legible. They all hang on the `query_search.branch` span (set
# from inside the entity-flow executor, which runs under that span, so
# `trace.get_current_span()` there IS the branch span) — except a few genuine
# sub-operation spans below. Attributes stay FLAT under the query_search root
# with a `branch_` prefix (matching branch_type / branch_uses_original_text);
# `branch_type` is the flow discriminator, so a similarity-only attr simply
# isn't set on a studio branch. Two groups earn a dot (rule C, >=2 siblings):
# `branch_weave_targets` and `branch_source`.
#
# Universal skeleton (every entity flow). `branch_entities` is the single
# consolidated identity; `branch_entity_resolved_counts` is index-aligned with
# it (per-entity PRE-union resolved candidate count — a 0 marks a silent drop);
# `branch_unresolved_entity_count` is the low-cardinality, label-eligible scalar
# version. `branch_result_count` is the post-union/post-limit hydrated total.
# The empty-result span EVENT message is a call-site string, not a Name.
QUERY_SEARCH_BRANCH_ENTITIES = QUERY_SEARCH.child("branch_entities")          # str[]
QUERY_SEARCH_BRANCH_ENTITY_RESOLVED_COUNTS = QUERY_SEARCH.child(
    "branch_entity_resolved_counts"
)                                                                            # int[]
QUERY_SEARCH_BRANCH_UNRESOLVED_ENTITY_COUNT = QUERY_SEARCH.child(
    "branch_unresolved_entity_count"
)                                                                            # int
QUERY_SEARCH_BRANCH_RESULT_COUNT = QUERY_SEARCH.child("branch_result_count")  # int
# `branch_aliases`: the LLM-expanded surface forms used for resolution. Set by
# studio (brand + freeform names) and non_character_franchise (expanded
# franchise_names). character_franchise splits its two form lists into the
# dedicated arrays below instead.
QUERY_SEARCH_BRANCH_ALIASES = QUERY_SEARCH.child("branch_aliases")           # str[]
# `branch_top_tier` / `branch_top_tier_count`: the top populated prominence
# tier and its size — set by the bucketed/tiered flows (person, both
# franchises). The name of the top tier is a closed low-cardinality set;
# the count is a measure.
QUERY_SEARCH_BRANCH_TOP_TIER = QUERY_SEARCH.child("branch_top_tier")          # str
QUERY_SEARCH_BRANCH_TOP_TIER_COUNT = QUERY_SEARCH.child(
    "branch_top_tier_count"
)                                                                            # int

# person: one child span per named person (parallel resolution across role
# tables), so per-person resolution latency is visible and its Postgres calls
# nest correctly. No aliases (LLM-free flow).
QUERY_SEARCH_PERSON_RESOLUTION = QUERY_SEARCH.child("person_resolution")      # span

# similarity: the branch span is organized around four questions a reader asks —
# (1) which traits mattered, (2) which avenues fetched candidates and how many
# each returned, (3) how strong the scoring weights were, (4) which paths were
# active in the final weave. Map-shaped signals are emitted as single JSON-string
# attributes (label + number side by side) rather than index-aligned parallel
# arrays: OTel span attributes can't hold a dict (it's dropped), but a JSON string
# is kept and renders readably in Tempo/Grafana. Tradeoff: no numeric TraceQL
# filter on an individual key inside the JSON — acceptable, these are for reading
# traces, not per-lane alerting. Plus a manual Qdrant span (the gRPC
# auto-instrumentation gap) around the shape-search `query_points` call.
#
# Single vs multi split: `branch_shape_modifiers` + scalar `branch_anchor_shape`
# are single-anchor only (the one anchor's additive weight deltas + its
# reach×quality shape); `branch_anchor_shape_cohesion` / `branch_lane_cohesion` /
# `branch_vector_space_cohesion` are multi-anchor only (per-shape / per-lane /
# per-vector-space cohort agreement). Everything else is set in both flows.

# (1) Traits marked important.
# Single-anchor: the additive lane-weight-delta anchor types that were enacted
# (cult_garbage / prestige / franchise_dominant / source_material — NOT
# standard_shape / studio_lineage / director_signature, which carry no delta).
# Always set, `"[]"` when none.
QUERY_SEARCH_BRANCH_SHAPE_MODIFIERS = QUERY_SEARCH.child(
    "branch_shape_modifiers"
)                                                                            # json array
# Single-anchor reach×quality shape bucket, scalar; `"none"` when shapeless (the
# common reception-50–80, sub-100K-reach middle cell).
QUERY_SEARCH_BRANCH_ANCHOR_SHAPE = QUERY_SEARCH.child("branch_anchor_shape")  # str
# Multi-anchor cohort shape composition {shape: M_s/N}, with a `"none"` key for
# the shapeless fraction so it sums to 1.
QUERY_SEARCH_BRANCH_ANCHOR_SHAPE_COHESION = QUERY_SEARCH.child(
    "branch_anchor_shape_cohesion"
)                                                                            # json map
# Multi-anchor per-metadata-lane cohort cohesion {lane: cohesion}.
QUERY_SEARCH_BRANCH_LANE_COHESION = QUERY_SEARCH.child(
    "branch_lane_cohesion"
)                                                                            # json map
# Multi-anchor per-vector-space cohort cohesion {space: cohesion}.
QUERY_SEARCH_BRANCH_VECTOR_SPACE_COHESION = QUERY_SEARCH.child(
    "branch_vector_space_cohesion"
)                                                                            # json map

# (2) Candidate-fetch avenues. {lane: result_count} for every retrieval query that
# actually ran (seed non-empty); a fired-but-empty lane is present at 0, a gated-off
# lane is absent. `branch_retrieval_total` is the deduped union size.
QUERY_SEARCH_BRANCH_RETRIEVAL_LANES = QUERY_SEARCH.child(
    "branch_retrieval_lanes"
)                                                                            # json map
QUERY_SEARCH_BRANCH_RETRIEVAL_TOTAL = QUERY_SEARCH.child(
    "branch_retrieval_total"
)                                                                            # int

# (3) Scoring weight strengths. {lane: normalized_weight} for the additive lanes,
# {space: weight} for the 8-space mix inside the shape lane.
QUERY_SEARCH_BRANCH_LANE_WEIGHTS = QUERY_SEARCH.child(
    "branch_lane_weights"
)                                                                            # json map
QUERY_SEARCH_BRANCH_VECTOR_SPACE_WEIGHTS = QUERY_SEARCH.child(
    "branch_vector_space_weights"
)                                                                            # json map

# (4) Final-weave paths. {bucket: target_slots} — the DESIRED per-bucket allocation
# `_compute_bucket_targets` set BEFORE weaving (best_overall's floor + any signal
# bucket that cleared the instantiation threshold), NOT the realized draw. A signal
# bucket absent from the map didn't instantiate; multi-bucket credit means an
# instantiated bucket can still draw 0 actual seats (its films entered via
# best_overall). `branch_low_cohesion_fallback` (multi-anchor centroid was noise →
# per-anchor round-robin; trivially false single-anchor).
QUERY_SEARCH_BRANCH_WEAVE_TARGETS = QUERY_SEARCH.child("branch_weave_targets")  # json map
QUERY_SEARCH_BRANCH_LOW_COHESION_FALLBACK = QUERY_SEARCH.child(
    "branch_low_cohesion_fallback"
)                                                                            # bool
# Multiplier/boost paths not recoverable from the additive weights or the fetch
# map — currently just `director_signature` (the auteur multiplier: the director
# lane is weight-0 and fires on any director, so neither lane_weights nor the
# retrieval map can reveal an active auteur boost). Omitted entirely when empty.
QUERY_SEARCH_BRANCH_ADDITIONAL_BOOSTS = QUERY_SEARCH.child(
    "branch_additional_boosts"
)                                                                            # json array

# similarity Qdrant probes: closes the gRPC auto-instrumentation gap for the
# similarity flow's TWO Qdrant calls, discriminated by `probe_kind` (rule E value
# lives in `SimilarityQdrantProbeKind` in similar_movies.py) exactly like
# `semantic_qdrant` splits its three primitives under one name:
#   - `anchor_vectors` — the `retrieve` that loads the anchors' stored vectors
#     (records `requested_count` / `returned_count`; returned < requested = an
#     anchor absent from Qdrant, a silent gap).
#   - `shape` — the batched named-vector probe (`query_batch_points`). It queries N
#     spaces in ONE call, so instead of semantic's single `vector_space` it carries
#     `space_count` + `spaces` (JSON list), `limit_per_space` (surfaces the 2×
#     over-fetch under a filter), `filter_active`, `hit_count` (total across spaces)
#     and `hits_by_space` (JSON {space:count}, per-space recall — a space starved by
#     the filter shows here). Leaf vocab (`probe_kind`/`filter_active`/`hit_count`)
#     matches `semantic_qdrant` so a reader query spans both.
QUERY_SEARCH_SIMILARITY_QDRANT = QUERY_SEARCH.child("similarity_qdrant")      # span
QUERY_SEARCH_SIMILARITY_QDRANT_PROBE_KIND = QUERY_SEARCH_SIMILARITY_QDRANT.child(
    "probe_kind"
)                                                                            # str
QUERY_SEARCH_SIMILARITY_QDRANT_REQUESTED_COUNT = QUERY_SEARCH_SIMILARITY_QDRANT.child(
    "requested_count"
)                                                                            # int (anchor_vectors)
QUERY_SEARCH_SIMILARITY_QDRANT_RETURNED_COUNT = QUERY_SEARCH_SIMILARITY_QDRANT.child(
    "returned_count"
)                                                                            # int (anchor_vectors)
QUERY_SEARCH_SIMILARITY_QDRANT_SPACE_COUNT = QUERY_SEARCH_SIMILARITY_QDRANT.child(
    "space_count"
)                                                                            # int (shape)
QUERY_SEARCH_SIMILARITY_QDRANT_SPACES = QUERY_SEARCH_SIMILARITY_QDRANT.child(
    "spaces"
)                                                                            # json array (shape)
QUERY_SEARCH_SIMILARITY_QDRANT_LIMIT_PER_SPACE = QUERY_SEARCH_SIMILARITY_QDRANT.child(
    "limit_per_space"
)                                                                            # int (shape)
QUERY_SEARCH_SIMILARITY_QDRANT_FILTER_ACTIVE = QUERY_SEARCH_SIMILARITY_QDRANT.child(
    "filter_active"
)                                                                            # bool (shape)
QUERY_SEARCH_SIMILARITY_QDRANT_HIT_COUNT = QUERY_SEARCH_SIMILARITY_QDRANT.child(
    "hit_count"
)                                                                            # int (shape)
QUERY_SEARCH_SIMILARITY_QDRANT_HITS_BY_SPACE = QUERY_SEARCH_SIMILARITY_QDRANT.child(
    "hits_by_space"
)                                                                            # json map (shape)

# similarity per-lane candidate fetch: one span per Postgres retrieval lane that
# actually ran (director / franchise / studio / source / quality / themes_recall /
# rare_medium). The qdrant shape probe is deliberately NOT wrapped — its params
# never vary and it already has `similarity_qdrant`. Each span names the `lane` and
# records the concrete `match` values the lane queried on (the bound IN-list IDs the
# auto-instrumented SQL span parameterizes away, so a reader can see WHY a lane
# returned what it did) plus its `result_count`. `similarity_fetch` earns a dotted
# namespace (rule C: it groups >= 2 emitted attributes). `match` is a JSON-string
# object because its keys vary by lane (lineage_entry_ids for franchise, company_ids
# for studio, bucket+limit for quality, …) — OTel can't hold a dict, a JSON string
# renders readably; the ID lists inside are high-cardinality span-only values.
QUERY_SEARCH_SIMILARITY_FETCH = QUERY_SEARCH.child("similarity_fetch")        # span
QUERY_SEARCH_SIMILARITY_FETCH_LANE = QUERY_SEARCH_SIMILARITY_FETCH.child(
    "lane"
)                                                                            # str
QUERY_SEARCH_SIMILARITY_FETCH_MATCH = QUERY_SEARCH_SIMILARITY_FETCH.child(
    "match"
)                                                                            # json map
QUERY_SEARCH_SIMILARITY_FETCH_RESULT_COUNT = QUERY_SEARCH_SIMILARITY_FETCH.child(
    "result_count"
)                                                                            # int

# --- query_search: semantic-endpoint Qdrant probes (1c-1 Bite 6) ---
# The gRPC auto-instrumentation gap for the SEMANTIC endpoint. One manual span
# per `query_points` primitive in
# search_v2/endpoint_fetching/semantic_query_execution.py (calibration probe,
# filtered pool probe, HasId reranker) — NO per-vector-space wrapper spans; the
# probe spans already render the fan-out, and double-wrapping would inflate the
# ~130-150 span/request budget. `semantic_qdrant` earns a dotted namespace
# (rule C): it groups five emitted attributes. All five are span-attr-only,
# never metric labels (rule F) — `probe_kind`/`vector_space`/`filter_active`
# are low-cardinality, `limit`/`hit_count` are measures.
#   probe_kind:    which primitive fired (QdrantProbeKind str-enum, owned by the
#                  call-site module per rule E) — calibration | pool | hasid_score.
#   vector_space:  the named vector queried (`using=vector_name.value`).
#   limit:         the `limit` arg (probe breadth; == pool size on the reranker).
#   filter_active: whether the USER HARD FILTER was applied. True only on the
#                  filtered pool probe; False on the unfiltered calibration probe
#                  and on the HasId reranker (its HasIdCondition is a pool
#                  restriction, not the hard filter). Reads as a state predicate,
#                  consistent with `step_1_unused` / `branch_low_cohesion_fallback`.
#   hit_count:     `len(response.points)` returned (rule D `_count` suffix) — the
#                  diagnostic that explains downstream elbow/pathology behavior.
#   query_params:  the JSON of the space body that produced this probe's query
#                  vector (`model_dump_json(exclude_defaults=True)` — only the
#                  populated fields that feed embedding_text). High-cardinality
#                  free text; span-attr-only, never a metric label (rule F).
QUERY_SEARCH_SEMANTIC_QDRANT = QUERY_SEARCH.child("semantic_qdrant")          # span
QUERY_SEARCH_SEMANTIC_QDRANT_PROBE_KIND = QUERY_SEARCH_SEMANTIC_QDRANT.child(
    "probe_kind"
)                                                                            # str
QUERY_SEARCH_SEMANTIC_QDRANT_VECTOR_SPACE = QUERY_SEARCH_SEMANTIC_QDRANT.child(
    "vector_space"
)                                                                            # str
QUERY_SEARCH_SEMANTIC_QDRANT_LIMIT = QUERY_SEARCH_SEMANTIC_QDRANT.child("limit")  # int
QUERY_SEARCH_SEMANTIC_QDRANT_FILTER_ACTIVE = QUERY_SEARCH_SEMANTIC_QDRANT.child(
    "filter_active"
)                                                                            # bool
QUERY_SEARCH_SEMANTIC_QDRANT_HIT_COUNT = QUERY_SEARCH_SEMANTIC_QDRANT.child(
    "hit_count"
)                                                                            # int
QUERY_SEARCH_SEMANTIC_QDRANT_QUERY_PARAMS = QUERY_SEARCH_SEMANTIC_QDRANT.child(
    "query_params"
)                                                                            # str (JSON)

# exact_title: `branch_exact_title_year` (set only when Step 0 supplied a year);
# `branch_source.*` — the result composition by RETRIEVAL MECHANISM (seed = exact
# title match, close = pg_trgm token-superset, fanout = franchise siblings,
# title_only = title matched but year didn't). seed/close/fanout are always-on
# (0-safe); title_only is CONDITIONAL (set only when a year was supplied —
# meaningless without one). `seed_count == 0` with results > 0 means nothing
# returned is an actual title match.
QUERY_SEARCH_BRANCH_EXACT_TITLE_YEAR = QUERY_SEARCH.child(
    "branch_exact_title_year"
)                                                                            # int
QUERY_SEARCH_BRANCH_SOURCE = QUERY_SEARCH.child("branch_source")              # ns
QUERY_SEARCH_BRANCH_SOURCE_SEED_COUNT = QUERY_SEARCH_BRANCH_SOURCE.child(
    "seed_count"
)                                                                            # int
QUERY_SEARCH_BRANCH_SOURCE_CLOSE_COUNT = QUERY_SEARCH_BRANCH_SOURCE.child(
    "close_count"
)                                                                            # int
QUERY_SEARCH_BRANCH_SOURCE_FANOUT_COUNT = QUERY_SEARCH_BRANCH_SOURCE.child(
    "fanout_count"
)                                                                            # int
QUERY_SEARCH_BRANCH_SOURCE_TITLE_ONLY_COUNT = QUERY_SEARCH_BRANCH_SOURCE.child(
    "title_only_count"
)                                                                            # int

# studio: `branch_studio_llm_fallback` (flow-level — the whole translation call
# failed and every ref degraded to freeform); `branch_studio_entity_paths`
# (per-ref "brand"/"freeform", index-aligned with branch_entities — normal on a
# successful call, NOT a degradation); `branch_studio_brand_names`; brand/freeform
# match counts (from the already-separate brand/freeform paths in _execute_any).
# Brand refs present with brand_match_count == 0 is the silent dead-end — brand
# wins per ref with NO fall-through to freeform.
QUERY_SEARCH_BRANCH_STUDIO_LLM_FALLBACK = QUERY_SEARCH.child(
    "branch_studio_llm_fallback"
)                                                                            # bool
QUERY_SEARCH_BRANCH_STUDIO_ENTITY_PATHS = QUERY_SEARCH.child(
    "branch_studio_entity_paths"
)                                                                            # str[]
QUERY_SEARCH_BRANCH_STUDIO_BRAND_NAMES = QUERY_SEARCH.child(
    "branch_studio_brand_names"
)                                                                            # str[]
QUERY_SEARCH_BRANCH_STUDIO_BRAND_MATCH_COUNT = QUERY_SEARCH.child(
    "branch_studio_brand_match_count"
)                                                                            # int
QUERY_SEARCH_BRANCH_STUDIO_FREEFORM_MATCH_COUNT = QUERY_SEARCH.child(
    "branch_studio_freeform_match_count"
)                                                                            # int

# non_character_franchise: `branch_franchise_llm_fallback` (LLM alias expansion
# failed -> [canonical_name]); `branch_secondary_count` (universe-only bucket).
# Aliases -> branch_aliases; top tier (primary/lineage) -> branch_top_tier[_count].
QUERY_SEARCH_BRANCH_FRANCHISE_LLM_FALLBACK = QUERY_SEARCH.child(
    "branch_franchise_llm_fallback"
)                                                                            # bool
QUERY_SEARCH_BRANCH_SECONDARY_COUNT = QUERY_SEARCH.child(
    "branch_secondary_count"
)                                                                            # int

# character_franchise: `branch_character_franchise_llm_failed` (the fanout has NO
# fallback — failure -> empty result, so this distinguishes "LLM died" from
# "catalog gap"); the two form lists split into dedicated arrays; per-tier counts
# (7 tiers). Two child spans wrap the parallel franchise/character resolutions
# (the sequential lineage-mainline split folds into the franchise span). Top tier
# (tier_1) -> branch_top_tier[_count].
QUERY_SEARCH_BRANCH_CHARACTER_FRANCHISE_LLM_FAILED = QUERY_SEARCH.child(
    "branch_character_franchise_llm_failed"
)                                                                            # bool
QUERY_SEARCH_BRANCH_CHARACTER_FORMS = QUERY_SEARCH.child(
    "branch_character_forms"
)                                                                            # str[]
QUERY_SEARCH_BRANCH_FRANCHISE_FORMS = QUERY_SEARCH.child(
    "branch_franchise_forms"
)                                                                            # str[]
QUERY_SEARCH_BRANCH_TIER_COUNTS = QUERY_SEARCH.child("branch_tier_counts")    # int[]
QUERY_SEARCH_FRANCHISE_RESOLUTION = QUERY_SEARCH.child(
    "franchise_resolution"
)                                                                            # span
QUERY_SEARCH_CHARACTER_RESOLUTION = QUERY_SEARCH.child(
    "character_resolution"
)                                                                            # span

# --- filters: cross-endpoint hard-filter input attributes (raw wire values) ---
# `filters` earns a root (rule C): eleven emitted siblings, and the same wire
# shape (MetadataFiltersInput) is shared by /query_search today and
# /rerun_query_search, /similarity_search, /attribute_search when they are
# instrumented (1c-2..4). One attr per wire field, set ONLY when the client
# sent that field — attribute existence answers "is this filter active?",
# the typed value carries the debugging detail. Values are captured
# PRE-translation (the raw enum strings, not resolved enums / offer keys) so
# a translation failure still shows exactly what the client sent. Per-field
# attrs are high-cardinality-OK span attrs (rule F); `active_count` is the
# one low-cardinality, label-eligible member (always set; 0 = no filters).
FILTERS = Name("filters")
FILTERS_MIN_RELEASE_TS = FILTERS.child("min_release_ts")
FILTERS_MAX_RELEASE_TS = FILTERS.child("max_release_ts")
FILTERS_MIN_RUNTIME = FILTERS.child("min_runtime")
FILTERS_MAX_RUNTIME = FILTERS.child("max_runtime")
FILTERS_MIN_MATURITY_RANK = FILTERS.child("min_maturity_rank")
FILTERS_MAX_MATURITY_RANK = FILTERS.child("max_maturity_rank")
FILTERS_GENRES = FILTERS.child("genres")
FILTERS_AUDIO_LANGUAGES = FILTERS.child("audio_languages")
FILTERS_KEYWORDS = FILTERS.child("keywords")
FILTERS_STREAMING_SERVICES = FILTERS.child("streaming_services")
FILTERS_ACTIVE_COUNT = FILTERS.child("active_count")

# --- llm: the shared LLM-router span + the facts OTel has no standard key for ---
# `generate_llm_response_async` is the single codepath every LLM call passes
# through, so one `llm.generate` span (wrapping its whole retry loop) covers
# every step's call; step identity comes from the parent span nesting, never
# duplicated here. This root earns a dot (rule C): it groups >= 2 emitted
# attributes (attempt_count, cost_usd, prompt_version) beyond the span name.
#
# Standard facts — provider, model, token usage (input/output AND cached reads)
# — are OTel GenAI semantic-convention keys (`gen_ai.system`,
# `gen_ai.request.model`, `gen_ai.usage.*`, including
# `gen_ai.usage.cache_read.input_tokens`) and the standard `error.type`; those
# are emitted at the call site as the spec's own strings and are deliberately
# NOT authored here (a standard root is never re-spelled in this registry).
# This root owns only what the GenAI conventions have no key for:
#   - cost_usd:        computed dollar cost (rule D unit suffix; low-card? no —
#                      a continuous measure, span-attr-only, never a metric label)
#   - prompt_version:  short content hash of the SYSTEM prompt — changes iff the
#                      prompt text changes; lets evals slice by prompt revision
#   - attempt_count:   attempts made (1 = clean first try; > 1 = retried). With
#                      the span status this separates clean / recovered /
#                      exhausted (see query_search_planning.md §2.8). Low-card,
#                      metric-label-eligible.
# Cached input tokens are NOT here: the GenAI semconv now has a stable key,
# `gen_ai.usage.cache_read.input_tokens` (tokens served from the provider's
# prompt cache — a subset of gen_ai.usage.input_tokens, billed at a discount),
# so the call site emits that standard string instead.
# The `llm.retry` / `llm.payload` span-event *messages* are human-readable
# strings (per this module's scope note), defined at the call site, not Names.
LLM = Name("llm")
LLM_GENERATE = LLM.child("generate")            # span name (the router span)
LLM_COST_USD = LLM.child("cost_usd")            # attr: computed dollar cost
LLM_PROMPT_VERSION = LLM.child("prompt_version")  # attr: system-prompt content hash
LLM_ATTEMPT_COUNT = LLM.child("attempt_count")  # attr: attempts made (>=1)

# --- embedding: the shared embedding-router span + the facts OTel has no key for ---
# `generate_vector_embedding` is the single codepath every embedding call passes
# through (search AND offline ingestion), so one `embedding.generate` span covers
# every embedding on every endpoint — the exact parallel to `llm.generate` for the
# LLM router. This root earns a dot (rule C): it groups >= 2 emitted attributes
# (cost_usd, input_count) beyond the span name.
#
# Standard facts — provider, model, operation, input token usage — are OTel GenAI
# semantic-convention keys (`gen_ai.system`, `gen_ai.request.model`,
# `gen_ai.operation.name`, `gen_ai.usage.input_tokens`) and the standard
# `error.type`; those are emitted at the call site as the spec's own strings and
# are deliberately NOT authored here (a standard root is never re-spelled in this
# registry). Embeddings produce no output tokens and have no prompt caching, so
# `gen_ai.usage.output_tokens` / `.cache_read.input_tokens` are never emitted (an
# always-0 key is noise). Unlike `llm.generate`, there is no retry loop (the
# embedding client runs max_retries=0), so there is no `attempt_count` sibling.
# This root owns only what the GenAI conventions have no key for:
#   - cost_usd:     computed dollar cost of the batch call (rule D unit suffix; a
#                   continuous measure — span-attr-only, never a metric label,
#                   rule F). Mirrors `llm.cost_usd`.
#   - input_count:  number of texts in the batch (a single call embeds up to 2048
#                   inputs), so cost/tokens-per-input is derivable. Rule D `_count`
#                   suffix; low-cardinality, metric-label-eligible.
EMBEDDING = Name("embedding")
EMBEDDING_GENERATE = EMBEDDING.child("generate")        # span name (the embedding router span)
EMBEDDING_COST_USD = EMBEDDING.child("cost_usd")        # attr: computed dollar cost (batch)
EMBEDDING_INPUT_COUNT = EMBEDDING.child("input_count")  # attr: number of texts in the batch
