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
     request.failure_reason  -> dot: `request` already groups >= 2 emitted
                                attributes (success, failure_reason, cost_usd, …)
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
   (`movie.payload_source`, `request.success`, `request.failure_reason`) are
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


# --- request: per-request rollups + verdict, on any endpoint's server span ---
# The transport/pipeline-agnostic "what did this request cost / return" facts,
# written on the FastAPI server span at handler (or stream) end. A cross-endpoint
# root (rule B): the owner is the request itself, not any single endpoint, so an
# endpoint-specific root would be wrong. Not every endpoint emits
# every leaf — a no-LLM endpoint (similarity_search, attribute_search,
# title_search) simply omits `cost_usd` / `usage.*`; absence == "spent nothing".
# The measures (`cost_usd`, `result_count`, `usage.*`) are continuous —
# span-attr-only, never metric labels (rule F). The verdict pair (`success` /
# `failure_reason`) is the exception: both are low-cardinality, so both stay
# metric-label-eligible.
#
# `success`: the request verdict bool, set on every path (exception: streaming
# endpoints under `record_outcome(success_on_return=False)` write it only on
# failure — see `/query_search`). `failure_reason`: names WHY, set only when
# `success` is false (a `FailureReason` value). Both live on the `request` root —
# the verdict is a per-request fact, not endpoint-specific (rule B) — and `request`
# earns their dot the same way it groups the rollups.
# `cost_usd`: summed USD across every LLM + embedding call the request incurred
# (all billed attempts, including retried/failed ones — see
# observability/cost_tracking.py). Flat leaf with the established `_usd` unit
# suffix (mirroring `llm.cost_usd`).
# `result_count`: the number of results the client received. Semantic nuance —
# on the streaming branch-plan endpoints (query_search / rerun_query_search) it
# is the pre-dedup SUM across branches (there is no server-side cross-branch
# merge); on the single-response endpoints it is the hydrated card count. Both
# read as "results the client received."
# `usage.*`: summed token usage across every LLM + embedding call (all billed
# attempts, mirroring cost). `usage` earns a dot (rule C): three emitted siblings
# mirroring the standard `gen_ai.usage.*` triad. `cached_input_tokens` is a
# SUBSET of `input_tokens` (cached ⊆ input), never additive to it.
REQUEST = Name("request")
REQUEST_SUCCESS = REQUEST.child("success")                           # request.success (bool)
REQUEST_FAILURE_REASON = REQUEST.child("failure_reason")             # request.failure_reason
REQUEST_COST_USD = REQUEST.child("cost_usd")                         # float
REQUEST_RESULT_COUNT = REQUEST.child("result_count")                 # int
REQUEST_USAGE = REQUEST.child("usage")
REQUEST_USAGE_INPUT_TOKENS = REQUEST_USAGE.child("input_tokens")
REQUEST_USAGE_CACHED_INPUT_TOKENS = REQUEST_USAGE.child("cached_input_tokens")
REQUEST_USAGE_OUTPUT_TOKENS = REQUEST_USAGE.child("output_tokens")

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
# The delivered-card count is the cross-endpoint `request.result_count` (above).
# `fuzzy_result_count` stays endpoint-owned — it's the typo/gap signal specific
# to the trigram fallback, not a generic per-request rollup.
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
# Request-level cost / token / result-count rollups moved to the cross-endpoint
# `request.*` root (REQUEST_COST_USD / REQUEST_USAGE_* / REQUEST_RESULT_COUNT
# above): they mean the same thing on every endpoint, so the owner is the request,
# not query_search. The branch counts stay here — "branch" is a branch-plan
# concept only query_search and /rerun_query_search have (rerun REUSES these keys,
# the same rule-B move as reusing the branch spans). They back the
# `request.success` verdict (success = >= 1 branch executed without a
# branch_error — see the `QUERY_SEARCH_BRANCH_ERROR` per-branch attribute
# below). Low-cardinality counts, metric-label-eligible.
QUERY_SEARCH_SUCCEEDED_BRANCH_COUNT = QUERY_SEARCH.child("succeeded_branch_count")  # int
QUERY_SEARCH_FAILED_BRANCH_COUNT = QUERY_SEARCH.child("failed_branch_count")        # int

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
# Group-A cost leaf. Flat `step_2_*` form to sit beside the existing step_2
# attrs (this span predates the dotted-namespace Stage-4 style). Sums the
# Step-2 LLM call's USD cost, mirroring the request-level `query_search.cost_usd`
# and per-call `llm.cost_usd` unit suffix.
QUERY_SEARCH_STEP_2_COST_USD = QUERY_SEARCH.child("step_2_cost_usd")  # attr (float)

# decomposition (Group B) — one span per branch bracketing the whole Step-3
# fan-out: every per-trait `trait` span (Step 3 + query generation) plus the
# sibling `implicit_expectations` span nest under it. Earns a dotted namespace
# (rule C). `decomposition.cost_usd` is the group's LLM cost rollup (Step 3 +
# handler query-generation + implicit-expectations calls).
QUERY_SEARCH_DECOMPOSITION = QUERY_SEARCH.child("decomposition")      # span + ns
QUERY_SEARCH_DECOMPOSITION_COST_USD = QUERY_SEARCH_DECOMPOSITION.child(
    "cost_usd"
)                                                                    # attr (float)

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
#   branch_error:             set ONLY when the branch soft-failed (Step 2 / Step 3
#                             / Stage 4 exhausted its LLM retries, or an unexpected
#                             escape) — the `repr(exc)` / error string carried in
#                             the terminal branch_results event. A DEGRADATION, not
#                             a span error: the branch span stays UNSET and the
#                             request verdict is untouched (the nested `llm.generate`
#                             child carries the ERROR status). Mirrors
#                             `trait_step_3_error`. High-cardinality → span-attr-only.
#   branch_cost_usd:          summed USD (LLM + embedding, all billed attempts)
#                             incurred inside this branch. ALWAYS set — 0.0 for
#                             entity flows and other cost-free branches (their LLM
#                             resolution happens in Step 0, before branch spans
#                             exist). Continuous measure → span-attr-only, never a
#                             metric label (rule F); `_usd` unit suffix mirroring
#                             `step_2_cost_usd` / `llm.cost_usd`. Note
#                             `sum(branch_cost_usd)` ≈ `request.cost_usd` minus the
#                             pre-branch Step 0 / Step 1 routing cost. Flat leaf
#                             (like `step_2_cost_usd`), not dotted — `branch` is a
#                             span-name leaf whose attrs are flat `branch_*`.
QUERY_SEARCH_BRANCH = QUERY_SEARCH.child("branch")                    # span name
QUERY_SEARCH_BRANCH_TYPE = QUERY_SEARCH.child("branch_type")          # attr (str)
QUERY_SEARCH_BRANCH_USES_ORIGINAL_TEXT = QUERY_SEARCH.child(
    "branch_uses_original_text"
)                                                                    # attr (bool)
QUERY_SEARCH_BRANCH_ERROR = QUERY_SEARCH.child("branch_error")        # attr (str)
QUERY_SEARCH_BRANCH_COST_USD = QUERY_SEARCH.child(
    "branch_cost_usd"
)                                                                    # attr (float)

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

# --- person: flow-neutral per-person resolution span (shared engine) ---
# One person → prominence-bucket resolution (`fetch_person_buckets` in
# search_v2/person_search.py) is reached two ways — the /query_search person
# BRANCH (`run_person_search`, which resolves Step 0's named references) and the
# pure /attribute_search ENDPOINT (caller-supplied names, no LLM). The span +
# attributes below are produced INSIDE the shared wrapper `resolve_person_traced`,
# so they belong to neither caller: they hang off a `person` root (rule B — the
# domain that conceptually owns the work), NOT `query_search`. On the branch path
# the span lands under the `query_search.branch` span (the current span while the
# wrapper runs); on the endpoint path it lands under the FastAPI server span.
# Caller-specific facts stay on their own roots: the branch keeps its
# `query_search.branch_*` aggregate skeleton (branch_entities, the index-aligned
# branch_entity_resolved_counts, branch_top_tier); the endpoint adds
# `attribute_search.*` (see the ATTRIBUTE_SEARCH root below). This mirrors the
# SIMILARITY split above.
#
# `resolve` earns a dotted namespace (rule C: it groups >= 2 emitted attributes).
# `name` is high-cardinality → span-attr-only (rule F); `movie_count` is a measure
# (rule D `_count` suffix), 0 == the person resolved to no filter-eligible credits
# (the silent-drop signal, also flagged by the `"person unresolved"` span event);
# `best_bucket` is the most-prominent bucket reached (1=lead … 4=minor), omitted
# when movie_count is 0 — an all-bucket-4 result for a well-known name is the
# fingerprint of a resolution collision.
PERSON = Name("person")
PERSON_RESOLVE = PERSON.child("resolve")                                     # span
PERSON_RESOLVE_NAME = PERSON_RESOLVE.child("name")                           # str
PERSON_RESOLVE_MOVIE_COUNT = PERSON_RESOLVE.child("movie_count")             # int
PERSON_RESOLVE_BEST_BUCKET = PERSON_RESOLVE.child("best_bucket")             # int 1-4

# --- similarity: flow-neutral engine spans + signal attributes (1c-3) ---
# The similar-movies engine (`run_similar_movies_for_ids` in
# search_v2/similar_movies.py) is reached two ways — the /query_search similarity
# BRANCH (`run_similarity_search`, which resolves NL reference titles to anchors)
# and the pure /similarity_search ENDPOINT (caller-supplied anchor IDs, no LLM). The
# spans and signal attributes below are produced INSIDE the shared engine, so they
# belong to neither caller: they hang off a `similarity` root (rule B — the domain
# that conceptually owns the work), NOT `query_search`. On the branch path they land
# on the `query_search.branch` span (the current span while the engine runs); on the
# endpoint path they land on the FastAPI server span. Caller-specific facts stay on
# their own roots: the branch keeps its `query_search.branch_*` attrs (branch_type,
# the reference-resolution entity skeleton, branch_result_count); the endpoint adds
# `similarity_search.*` (see the SIMILARITY_SEARCH root below).
SIMILARITY = Name("similarity")

# anchor_count: number of anchor movies driving the search — the single-vs-multi-
# anchor discriminator that selects the two engine pipelines and governs how every
# signal below reads. On the endpoint path it stands in for the query_search entity
# skeleton (which counted resolved reference titles); low-cardinality small int,
# label-eligible. Set by the engine, so both callers get it.
SIMILARITY_ANCHOR_COUNT = SIMILARITY.child("anchor_count")                   # int

# The signal attributes are organized around four questions a reader asks — (1)
# which traits mattered, (2) which avenues fetched candidates and how many each
# returned, (3) how strong the scoring weights were, (4) which paths were active in
# the final weave. Map-shaped signals are emitted as single JSON-string attributes
# (label + number side by side) rather than index-aligned parallel arrays: OTel span
# attributes can't hold a dict (it's dropped), but a JSON string is kept and renders
# readably in Tempo/Grafana. Tradeoff: no numeric TraceQL filter on an individual
# key inside the JSON — acceptable, these are for reading traces, not per-lane
# alerting.
#
# Single vs multi split: `shape_modifiers` + scalar `anchor_shape` are single-anchor
# only (the one anchor's additive weight deltas + its reach×quality shape);
# `anchor_shape_cohesion` / `lane_cohesion` / `vector_space_cohesion` are
# multi-anchor only (per-shape / per-lane / per-vector-space cohort agreement).
# Everything else is set in both flows.

# (1) Traits marked important.
# Single-anchor: the additive lane-weight-delta anchor types that were enacted
# (cult_garbage / prestige / franchise_dominant / source_material — NOT
# standard_shape / studio_lineage / director_signature, which carry no delta).
# Always set, `"[]"` when none.
SIMILARITY_SHAPE_MODIFIERS = SIMILARITY.child(
    "shape_modifiers"
)                                                                            # json array
# Single-anchor reach×quality shape bucket, scalar; `"none"` when shapeless (the
# common reception-50–80, sub-100K-reach middle cell).
SIMILARITY_ANCHOR_SHAPE = SIMILARITY.child("anchor_shape")                   # str
# Multi-anchor cohort shape composition {shape: M_s/N}, with a `"none"` key for
# the shapeless fraction so it sums to 1.
SIMILARITY_ANCHOR_SHAPE_COHESION = SIMILARITY.child(
    "anchor_shape_cohesion"
)                                                                            # json map
# Multi-anchor per-metadata-lane cohort cohesion {lane: cohesion}.
SIMILARITY_LANE_COHESION = SIMILARITY.child(
    "lane_cohesion"
)                                                                            # json map
# Multi-anchor per-vector-space cohort cohesion {space: cohesion}.
SIMILARITY_VECTOR_SPACE_COHESION = SIMILARITY.child(
    "vector_space_cohesion"
)                                                                            # json map

# (2) Candidate-fetch avenues. {lane: result_count} for every retrieval query that
# actually ran (seed non-empty); a fired-but-empty lane is present at 0, a gated-off
# lane is absent. `retrieval_total` is the deduped union size.
SIMILARITY_RETRIEVAL_LANES = SIMILARITY.child(
    "retrieval_lanes"
)                                                                            # json map
SIMILARITY_RETRIEVAL_TOTAL = SIMILARITY.child(
    "retrieval_total"
)                                                                            # int

# (3) Scoring weight strengths. {lane: normalized_weight} for the additive lanes,
# {space: weight} for the 8-space mix inside the shape lane.
SIMILARITY_LANE_WEIGHTS = SIMILARITY.child(
    "lane_weights"
)                                                                            # json map
SIMILARITY_VECTOR_SPACE_WEIGHTS = SIMILARITY.child(
    "vector_space_weights"
)                                                                            # json map

# (4) Final-weave paths. {bucket: target_slots} — the DESIRED per-bucket allocation
# `_compute_bucket_targets` set BEFORE weaving (best_overall's floor + any signal
# bucket that cleared the instantiation threshold), NOT the realized draw. A signal
# bucket absent from the map didn't instantiate; multi-bucket credit means an
# instantiated bucket can still draw 0 actual seats (its films entered via
# best_overall). `low_cohesion_fallback` (multi-anchor centroid was noise →
# per-anchor round-robin; trivially false single-anchor).
SIMILARITY_WEAVE_TARGETS = SIMILARITY.child("weave_targets")                 # json map
SIMILARITY_LOW_COHESION_FALLBACK = SIMILARITY.child(
    "low_cohesion_fallback"
)                                                                            # bool
# Multiplier/boost paths not recoverable from the additive weights or the fetch
# map — currently just `director_signature` (the auteur multiplier: the director
# lane is weight-0 and fires on any director, so neither lane_weights nor the
# retrieval map can reveal an active auteur boost). Omitted entirely when empty.
SIMILARITY_ADDITIONAL_BOOSTS = SIMILARITY.child(
    "additional_boosts"
)                                                                            # json array

# similarity Qdrant probes (`similarity.qdrant`): closes the gRPC
# auto-instrumentation gap for the engine's TWO Qdrant calls, discriminated by
# `probe_kind` (rule E value lives in `SimilarityQdrantProbeKind` in
# similar_movies.py) exactly like `semantic_qdrant` splits its three primitives
# under one name:
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
SIMILARITY_QDRANT = SIMILARITY.child("qdrant")                               # span
SIMILARITY_QDRANT_PROBE_KIND = SIMILARITY_QDRANT.child(
    "probe_kind"
)                                                                            # str
SIMILARITY_QDRANT_REQUESTED_COUNT = SIMILARITY_QDRANT.child(
    "requested_count"
)                                                                            # int (anchor_vectors)
SIMILARITY_QDRANT_RETURNED_COUNT = SIMILARITY_QDRANT.child(
    "returned_count"
)                                                                            # int (anchor_vectors)
SIMILARITY_QDRANT_SPACE_COUNT = SIMILARITY_QDRANT.child(
    "space_count"
)                                                                            # int (shape)
SIMILARITY_QDRANT_SPACES = SIMILARITY_QDRANT.child(
    "spaces"
)                                                                            # json array (shape)
SIMILARITY_QDRANT_LIMIT_PER_SPACE = SIMILARITY_QDRANT.child(
    "limit_per_space"
)                                                                            # int (shape)
SIMILARITY_QDRANT_FILTER_ACTIVE = SIMILARITY_QDRANT.child(
    "filter_active"
)                                                                            # bool (shape)
SIMILARITY_QDRANT_HIT_COUNT = SIMILARITY_QDRANT.child(
    "hit_count"
)                                                                            # int (shape)
SIMILARITY_QDRANT_HITS_BY_SPACE = SIMILARITY_QDRANT.child(
    "hits_by_space"
)                                                                            # json map (shape)

# similarity per-lane candidate fetch (`similarity.fetch`): one span per Postgres
# retrieval lane that actually ran (director / franchise / studio / source /
# quality / themes_recall / rare_medium). The qdrant shape probe is deliberately NOT
# wrapped — its params never vary and it already has `similarity.qdrant`. Each span
# names the `lane` and records the concrete `match` values the lane queried on (the
# bound IN-list IDs the auto-instrumented SQL span parameterizes away, so a reader
# can see WHY a lane returned what it did) plus its `result_count`. `fetch` earns a
# dotted namespace (rule C: it groups >= 2 emitted attributes). `match` is a
# JSON-string object because its keys vary by lane (lineage_entry_ids for franchise,
# company_ids for studio, bucket+limit for quality, …) — OTel can't hold a dict, a
# JSON string renders readably; the ID lists inside are high-cardinality span-only
# values.
SIMILARITY_FETCH = SIMILARITY.child("fetch")                                 # span
SIMILARITY_FETCH_LANE = SIMILARITY_FETCH.child(
    "lane"
)                                                                            # str
SIMILARITY_FETCH_MATCH = SIMILARITY_FETCH.child(
    "match"
)                                                                            # json map
SIMILARITY_FETCH_RESULT_COUNT = SIMILARITY_FETCH.child(
    "result_count"
)                                                                            # int

# --- similarity_search: endpoint-owned facts on the /similarity_search server span ---
# The pure endpoint (`api/main.py`) reuses the shared engine (so it inherits every
# `similarity.*` span/attr above), but two facts are known only to the handler and
# so are owned by the endpoint (rule B): whether the response was served from the
# Redis result cache (`cache_hit` — set True on the warm return, False on a cold
# compute; mirrors the movie.payload_source honesty rule by being set on both
# success paths). The delivered-card count is the cross-endpoint
# `request.result_count` (above), not an endpoint-owned leaf. `cache_hit` is
# low-cardinality and label-eligible.
SIMILARITY_SEARCH = Name("similarity_search")
SIMILARITY_SEARCH_CACHE_HIT = SIMILARITY_SEARCH.child("cache_hit")           # bool

# --- attribute_search: endpoint-owned facts on the /attribute_search server span ---
# The deterministic browse endpoint (`api/main.py`) reuses two shared engines — the
# neutral-prior seed fetch (no-people path) and the `person.resolve` wrapper (per
# supplied name, so it inherits every `person.*` span/attr above) — plus the shared
# `filters.*` request-input attributes. The facts below are the ones known only to
# this handler / its orchestrator (`run_attribute_search`), so they are owned by the
# endpoint (rule B). Leaves stay flat with `_` (mirroring the `query_search.branch_*`
# grouped-facts precedent, not a `people.*` sub-namespace): each is one measured fact
# and `people` won't sub-expand into its own emitted telemetry tree.
#   path:                   browse | people — which ranking path ran (low-card,
#                           label-eligible; the single discriminator for slicing).
#   people_requested_count: named people received (post blank-strip at the boundary);
#                           always set (0 on the browse path). Low-card measure.
#   people_names:           the raw supplied names (per-element truncated) — high-card,
#                           span-attr-only (rule F); set only when people were sent.
#   people_searched_count:  names actually resolved after normalize + dedupe (==
#                           number of `person.resolve` spans). requested - searched =
#                           blank/duplicate names dropped.
#   people_unresolved_count: searched names that resolved to zero credits — the
#                           silent-drop signal in aggregate (alertable). searched -
#                           unresolved = names that contributed to the pool.
#   pool_count:             union pool size before hydration (people path); 0 here with
#                           searched_count > 0 is the "empty pool" case (distinct from
#                           the all-blank return where searched_count == 0).
# The delivered-card count is the cross-endpoint `request.result_count` (above),
# not an endpoint-owned leaf.
ATTRIBUTE_SEARCH = Name("attribute_search")
ATTRIBUTE_SEARCH_PATH = ATTRIBUTE_SEARCH.child("path")                        # str
ATTRIBUTE_SEARCH_PEOPLE_REQUESTED_COUNT = ATTRIBUTE_SEARCH.child(
    "people_requested_count"
)                                                                            # int
ATTRIBUTE_SEARCH_PEOPLE_NAMES = ATTRIBUTE_SEARCH.child("people_names")        # str[]
ATTRIBUTE_SEARCH_PEOPLE_SEARCHED_COUNT = ATTRIBUTE_SEARCH.child(
    "people_searched_count"
)                                                                            # int
ATTRIBUTE_SEARCH_PEOPLE_UNRESOLVED_COUNT = ATTRIBUTE_SEARCH.child(
    "people_unresolved_count"
)                                                                            # int
ATTRIBUTE_SEARCH_POOL_COUNT = ATTRIBUTE_SEARCH.child("pool_count")            # int

# --- rerun_query_search: request-boundary input attributes on the server span ---
# /rerun_query_search replays a prior search's branch set with a fresh filter
# set, bypassing Steps 0/1, so it REUSES every shared query_search.* span (branch,
# step_2, Stage 4, …) and the cross-endpoint request.* / filters.*
# rollups. The attrs below are the facts unique to rerun: its input IS the
# replayed branch plan (there is no raw user query or clarification to capture,
# unlike query_search), so these are rerun's analog of the query_search raw-query
# capture — owned by the endpoint (rule B), written at handler entry from the raw
# wire body before translation so a rejected trace still carries its input.
#   branch_count:     total fetches replayed (int, low-cardinality, label-eligible).
#   branch_types:     the `type` tag of each replayed branch (standard /
#                     exact_title / similarity / … — closed set, str[]); the rerun
#                     analog of step_0_flows ("what shape was replayed").
#   standard_queries: the up-to-3 standard branch queries (str[], high-cardinality,
#                     span-attr-only per rule F). Entity anchor names are NOT
#                     duplicated here — they already land on the entity branch
#                     spans (Bite 8), and branch_types shows which flows fired.
RERUN_QUERY_SEARCH = Name("rerun_query_search")
RERUN_QUERY_SEARCH_BRANCH_COUNT = RERUN_QUERY_SEARCH.child("branch_count")    # int
RERUN_QUERY_SEARCH_BRANCH_TYPES = RERUN_QUERY_SEARCH.child("branch_types")    # str[]
RERUN_QUERY_SEARCH_STANDARD_QUERIES = RERUN_QUERY_SEARCH.child(
    "standard_queries"
)                                                                            # str[]

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

# --- query_search: implicit-prior spans (1c-1 Bite 4 [deferred] + Bite 7) ---
# The implicit-prior mechanism is a TWO-location flow, so it gets two spans,
# both per STANDARD branch (entity flows never run it), both under the
# query_search.branch span:
#
#   implicit_expectations (GENERATION) — brackets the policy LLM call
#     (search_v2/implicit_expectations.py) inside _run_implicit_expectations_for_branch.
#     No manual attributes of its own by design: the router's `llm.generate`
#     child already carries tokens/cost/prompt-hash/full payload, and the policy
#     OUTPUT is recorded on the application span below (beside what actually
#     fired). This span exists to name that llm.generate child's parent, time
#     the call (it sits on Stage 4's critical path — Stage 4 awaits it), and
#     anchor the `implicit_expectations_failed` soft-fail EVENT (its message +
#     error.type are call-site strings, not Names).
#
#   implicit_prior_rerank (APPLICATION) — brackets the post-Stage-4 rerank in
#     _apply_implicit_prior_rerank_for_branch: gate -> single-axis selection ->
#     Postgres signal fetch (which nests here) -> per-movie boost -> resort.
#     Earns a dotted namespace (rule C): it groups many emitted attributes.
#     Per ADR-087 the rerank is SINGLE-AXIS (popularity primary, quality
#     fallback), so `boost_axis` names the one axis that fired (or none). Both
#     priors' direction+strength are recorded so the LLM's proposal sits beside
#     the code's selection (caps + active flags) in ONE span — divergence
#     between "proposed" and "applied" is a single read. All attributes are
#     low-cardinality (label-eligible) except the two float caps and the
#     signal_missing_count measure (span-attr-only, rule F).
#       boost_axis:            which axis fired — popularity | quality | none
#                              (BoostAxis str-enum, rule E, owned by the rerank
#                              module).
#       popularity_direction / popularity_strength / quality_direction /
#       quality_strength:      the policy OUTPUT per axis (PriorDirection /
#                              PriorStrength values — none/positive/inverse,
#                              none/light/normal/strong).
#       popularity_cap / quality_cap:  the strength->boost-ceiling the code
#                              resolved (float; 0 disables the axis).
#       popularity_active / quality_active:  the selection variables — which axis
#                              the code chose to fire (bool predicates).
#       inverse_applied:       the fired axis used the inverse direction (rewards
#                              LOW popularity / LOW reception). False when
#                              boost_axis=none. Derivable from boost_axis + the
#                              direction attrs, recorded directly for legibility
#                              (cf. step_1_unused).
#       noop_reason:           set ONLY when boost_axis=none — why the prior did
#                              nothing (PriorNoopReason str-enum): policy_unavailable
#                              (generation soft-failed / no branch) | branch_error |
#                              empty_pool | both_axes_off. Disambiguates the four
#                              no-op causes the active flags alone can't (the first
#                              three return from the gate before caps/active exist).
#       signal_missing_count:  candidates whose FIRED-axis signal was NULL in
#                              Postgres (-> 0 boost for that movie): the
#                              data-coverage risk, since obscure films are the
#                              likeliest to lack popularity/reception. Rule D
#                              `_count` suffix.
QUERY_SEARCH_IMPLICIT_EXPECTATIONS = QUERY_SEARCH.child(
    "implicit_expectations"
)                                                                            # span
QUERY_SEARCH_IMPLICIT_PRIOR_RERANK = QUERY_SEARCH.child(
    "implicit_prior_rerank"
)                                                                            # span + ns
QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_BOOST_AXIS = QUERY_SEARCH_IMPLICIT_PRIOR_RERANK.child(
    "boost_axis"
)                                                                            # str
QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_POPULARITY_DIRECTION = QUERY_SEARCH_IMPLICIT_PRIOR_RERANK.child(
    "popularity_direction"
)                                                                            # str
QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_POPULARITY_STRENGTH = QUERY_SEARCH_IMPLICIT_PRIOR_RERANK.child(
    "popularity_strength"
)                                                                            # str
QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_QUALITY_DIRECTION = QUERY_SEARCH_IMPLICIT_PRIOR_RERANK.child(
    "quality_direction"
)                                                                            # str
QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_QUALITY_STRENGTH = QUERY_SEARCH_IMPLICIT_PRIOR_RERANK.child(
    "quality_strength"
)                                                                            # str
QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_POPULARITY_CAP = QUERY_SEARCH_IMPLICIT_PRIOR_RERANK.child(
    "popularity_cap"
)                                                                            # float
QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_QUALITY_CAP = QUERY_SEARCH_IMPLICIT_PRIOR_RERANK.child(
    "quality_cap"
)                                                                            # float
QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_POPULARITY_ACTIVE = QUERY_SEARCH_IMPLICIT_PRIOR_RERANK.child(
    "popularity_active"
)                                                                            # bool
QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_QUALITY_ACTIVE = QUERY_SEARCH_IMPLICIT_PRIOR_RERANK.child(
    "quality_active"
)                                                                            # bool
QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_INVERSE_APPLIED = QUERY_SEARCH_IMPLICIT_PRIOR_RERANK.child(
    "inverse_applied"
)                                                                            # bool
QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_NOOP_REASON = QUERY_SEARCH_IMPLICIT_PRIOR_RERANK.child(
    "noop_reason"
)                                                                            # str
QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_SIGNAL_MISSING_COUNT = QUERY_SEARCH_IMPLICIT_PRIOR_RERANK.child(
    "signal_missing_count"
)                                                                            # int

# --- query_search: Stage 4 execution spans (Phase B pool definition + Phase C/D
# reranker dispatch). All are children of `query_search.branch` and cover the
# execution that turns generated endpoint specs into a scored candidate pool.
# Each earns a dotted namespace (rule C): every span groups >= 2 emitted
# attributes, mirroring `semantic_qdrant`.
#
# `candidate_generation` (Group C) is the branch-level wrapper over the whole
# pool-definition phase — the `generators` / `promotion` / `neutral_seed` /
# `auxiliary_shorts_exclusion` spans below all nest under it. `rerankers`
# (Group D) now covers BOTH positive and negative reranker dispatch, run in
# parallel; polarity is recorded per dispatch (see `dispatch.polarity`), so
# there is no separate `negatives` span.
QUERY_SEARCH_CANDIDATE_GENERATION = QUERY_SEARCH.child(
    "candidate_generation"
)                                                                            # span + ns (Group C)
QUERY_SEARCH_CANDIDATE_GENERATION_FETCH_COUNT = QUERY_SEARCH_CANDIDATE_GENERATION.child(
    "fetch_count"
)                                                                            # int (dispatch calls issued, post-dedup)
QUERY_SEARCH_CANDIDATE_GENERATION_CANDIDATE_COUNT = QUERY_SEARCH_CANDIDATE_GENERATION.child(
    "candidate_count"
)                                                                            # int (deduped union size)
QUERY_SEARCH_CANDIDATE_GENERATION_COST_USD = QUERY_SEARCH_CANDIDATE_GENERATION.child(
    "cost_usd"
)                                                                            # float (semantic-generator embeddings)

QUERY_SEARCH_GENERATORS = QUERY_SEARCH.child("generators")                   # span + ns
QUERY_SEARCH_GENERATORS_RAW_UNION_COUNT = QUERY_SEARCH_GENERATORS.child(
    "raw_union_count"
)                                                                            # int (pre-shorts)
QUERY_SEARCH_GENERATORS_SHORTS_REMOVED_COUNT = QUERY_SEARCH_GENERATORS.child(
    "shorts_removed_count"
)                                                                            # int
QUERY_SEARCH_GENERATORS_FINAL_POOL_COUNT = QUERY_SEARCH_GENERATORS.child(
    "final_pool_count"
)                                                                            # int (post-shorts)

QUERY_SEARCH_PROMOTION = QUERY_SEARCH.child("promotion")                     # span + ns
QUERY_SEARCH_PROMOTION_TIER = QUERY_SEARCH_PROMOTION.child("tier")           # str (PromotionTier.name)
QUERY_SEARCH_PROMOTION_POOL_COUNT_BEFORE = QUERY_SEARCH_PROMOTION.child(
    "pool_count_before"
)                                                                            # int
QUERY_SEARCH_PROMOTION_PROMOTED_SPEC_COUNT = QUERY_SEARCH_PROMOTION.child(
    "promoted_spec_count"
)                                                                            # int
QUERY_SEARCH_PROMOTION_POOL_COUNT_AFTER = QUERY_SEARCH_PROMOTION.child(
    "pool_count_after"
)                                                                            # int
QUERY_SEARCH_PROMOTION_SHORTS_REMOVED_COUNT = QUERY_SEARCH_PROMOTION.child(
    "shorts_removed_count"
)                                                                            # int

QUERY_SEARCH_NEUTRAL_SEED = QUERY_SEARCH.child("neutral_seed")               # span + ns
QUERY_SEARCH_NEUTRAL_SEED_REASON = QUERY_SEARCH_NEUTRAL_SEED.child("reason") # str (NeutralSeedReason)
QUERY_SEARCH_NEUTRAL_SEED_SEED_COUNT = QUERY_SEARCH_NEUTRAL_SEED.child(
    "seed_count"
)                                                                            # int (0 = fetch failed)

# rerankers (Group D) — positive AND negative reranker dispatch, run in
# parallel against the finalized union. `call_count` spans both polarities'
# dispatches; `cost_usd` is the group's embedding cost (semantic rerankers).
QUERY_SEARCH_RERANKERS = QUERY_SEARCH.child("rerankers")                     # span + ns
QUERY_SEARCH_RERANKERS_CALL_COUNT = QUERY_SEARCH_RERANKERS.child("call_count")  # int (positive + negative dispatches)
QUERY_SEARCH_RERANKERS_POOL_COUNT = QUERY_SEARCH_RERANKERS.child("pool_count")  # int
QUERY_SEARCH_RERANKERS_COST_USD = QUERY_SEARCH_RERANKERS.child("cost_usd")      # float

QUERY_SEARCH_DISPATCH = QUERY_SEARCH.child("dispatch")                       # span + ns
QUERY_SEARCH_DISPATCH_ROUTE = QUERY_SEARCH_DISPATCH.child("route")           # str (EndpointRoute.value)
QUERY_SEARCH_DISPATCH_OPERATION_TYPE = QUERY_SEARCH_DISPATCH.child(
    "operation_type"
)                                                                            # str (OperationType.value)
# Positive vs negative reranker dispatch — both run as POOL_RERANKER, so
# `operation_type` alone can't tell them apart once they share the `rerankers`
# span. Closed value set (Polarity.value: positive | negative), safe as a
# future metric label (rule F). Defaults to positive on generator/promotion/
# shorts dispatches (already distinguished by `operation_type`).
QUERY_SEARCH_DISPATCH_POLARITY = QUERY_SEARCH_DISPATCH.child("polarity")     # str (Polarity.value)
QUERY_SEARCH_DISPATCH_WAS_PROMOTED = QUERY_SEARCH_DISPATCH.child("was_promoted")  # bool
QUERY_SEARCH_DISPATCH_RESULT_COUNT = QUERY_SEARCH_DISPATCH.child("result_count")  # int
# The committed query params the executor actually runs — the endpoint spec
# serialized to compact JSON with the LLM analysis/reasoning scaffolding
# (thinking / *_exploration / *_candidates / *_intent / request_overview /
# keyword `attributes` walk, …) stripped, so the span shows *what* was queried
# without the generation-assist noise or the movie-ID bloat the SQL child spans
# carry. High-cardinality span attr — never a metric label (rule F).
QUERY_SEARCH_DISPATCH_QUERY_PARAMS = QUERY_SEARCH_DISPATCH.child("query_params")  # str (JSON)
# The default shorts-exclusion fetch reuses the dispatch machinery (`_dispatch_call`)
# but gets its own span NAME so it reads as the auxiliary shorts exclusion in the
# waterfall rather than an anonymous `dispatch`. It carries the same `dispatch.*`
# attributes (route = media_type, result_count = shorts fetched, timeout handling).
QUERY_SEARCH_AUXILIARY_SHORTS_EXCLUSION = QUERY_SEARCH.child(
    "auxiliary_shorts_exclusion"
)                                                                            # span

# --- query_search: Stage 4 scoring/aggregation (Group E) + hydration (Group F)
# spans. Children of `query_search.branch`, siblings of the execution spans above.
# `scoring` covers the per-trait combine/weight fold + branch aggregation over
# the already-dispatched reranker scores (positive AND negative dispatch now runs
# in the `rerankers` span; scoring only consumes the score maps). The
# `implicit_prior_rerank` span nests under `scoring` — the post-relevance boost
# is part of turning the scored pool into the final ranked list. `hydration`
# covers the bulk movie_card lookup that turns ranked ids into display cards.
# Each earns a dotted namespace (rule C): >= 2 emitted attrs.
QUERY_SEARCH_SCORING = QUERY_SEARCH.child("scoring")                          # span + ns
QUERY_SEARCH_SCORING_TRAIT_WEIGHTS = QUERY_SEARCH_SCORING.child(
    "trait_weights"
)                                                                            # str (JSON array)
QUERY_SEARCH_SCORING_RANKED_COUNT = QUERY_SEARCH_SCORING.child("ranked_count")    # int
QUERY_SEARCH_SCORING_TOP_SCORE = QUERY_SEARCH_SCORING.child("top_score")          # float

QUERY_SEARCH_HYDRATION = QUERY_SEARCH.child("hydration")                      # span + ns
QUERY_SEARCH_HYDRATION_REQUESTED_COUNT = QUERY_SEARCH_HYDRATION.child(
    "requested_count"
)                                                                            # int
QUERY_SEARCH_HYDRATION_RETURNED_COUNT = QUERY_SEARCH_HYDRATION.child(
    "returned_count"
)                                                                            # int

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
