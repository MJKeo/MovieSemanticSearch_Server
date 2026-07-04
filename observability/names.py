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
