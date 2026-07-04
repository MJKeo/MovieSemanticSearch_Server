# Search V2 — Title-only typeahead lookup.
#
# Backs the `GET /title_search` endpoint: a deterministic, NLP-free
# title-search path used by the frontend's "pick similar-to" mode.
# Called on every debounced keystroke, so the contract is minimal —
# normalized title match against `public.movie_card.title_normalized`,
# tier-ranked, return the top `limit` movie_ids.
#
# No LLM, no vector search, no Redis. Single Postgres query covers
# both matching tiers and applies the same 80/20 popularity/reception
# prior `/attribute_search` uses as its within-tier sort.
#
# Matching tiers (highest first):
#   1. Token-prefix: the query is a prefix of any whitespace-delimited
#      token in the title. The title starts with the query OR contains
#      "<space><query>" — covers "dark" → "The Dark Knight" because
#      "Dark" is a token-prefix.
#   2. Substring: the query appears anywhere in the title but not at a
#      token boundary — "ark kni" → "The Dark Knight".
#
# Tier 1 always ranks above tier 2 regardless of popularity. Within
# each tier, the shorter normalized title wins — "John Wick" outranks
# "John Wick: Chapter 2" for query "john wi" because it carries less
# title material beyond the query. Length ties fall through to the
# same 80/20 popularity-reception blend `/attribute_search` uses
# (so the three "King Kong" remakes, all 9 chars, sort by popularity),
# with `movie_id DESC` as a final stable tie-break.
#
# Fuzzy fallback tier: when the exact tiered scan above underfills
# (≤ TITLE_SEARCH_EXACT_UNDERFILL_TRIGGER hits) and the normalized query
# is long enough (≥ TITLE_SEARCH_FUZZY_MIN_QUERY_CHARS), a second pg_trgm
# `word_similarity` pass backfills the dropdown with near-miss titles
# ("intersteller" → "Interstellar", "race witch mountain" → "Race to
# Witch Mountain"). It is kept strictly off the hot path: well-spelled
# queries that fill the dropdown from the exact tiers pay zero extra
# round trips, protecting the p50<20ms target. Fuzzy results are ranked
# by word_similarity (then the same shorter-title / 80-20 prior the exact
# tiers use) and appended BELOW the exact results, never interleaved.
#
# Wire-level validation (`q` non-empty after trim, `limit` in [1, 25])
# lives in api/main.py — this module sees the already-validated query.

from __future__ import annotations

from typing import NamedTuple

from db.postgres import (
    fetch_title_search_fuzzy_movie_ids,
    fetch_title_search_movie_ids,
)
from implementation.misc.helpers import normalize_string
from implementation.misc.sql_like import escape_like


class TitleSearchResult(NamedTuple):
    """Ranked title-search result plus the telemetry the caller reports.

    `movie_ids` is the ordered result the endpoint hydrates and returns.
    `fuzzy_count` is how many of those ids came from the fuzzy fallback
    tier (0 whenever the fuzzy pass didn't fire) — surfaced so the API
    layer can record it as a span attribute without re-deriving which
    tier each id came from (the tier classification is otherwise
    discarded after ranking). Kept as a plain NamedTuple so the search
    module stays free of any observability dependency.
    """

    movie_ids: list[int]
    fuzzy_count: int


# Hard cap on results. The endpoint exposes a per-request `limit` query
# param, which api/main.py clamps to this ceiling before calling in.
# 25 is enough to backstop typeahead pickers without inviting abuse.
TITLE_SEARCH_MAX_LIMIT = 25

# Default `limit` when the caller omits the query param. Matches the
# spec's "10" — small enough for a snappy dropdown, large enough to
# show meaningful disambiguation for multi-instance titles ("King
# Kong" 1933 / 1976 / 2005).
TITLE_SEARCH_DEFAULT_LIMIT = 10

# Server-side truncation for excessively long queries. The endpoint
# accepts any length but we cap the LIKE-pattern input to keep the
# trigram-index probe bounded. 100 chars is well above any real movie
# title (the longest title in our catalog is ~80 chars including
# subtitle) so this is purely defensive against accidental or
# malicious oversized inputs.
TITLE_SEARCH_QUERY_MAX_CHARS = 100

# Fuzzy fallback tier triggers only when the exact tiered scan returns at
# most this many hits — i.e. the dropdown is underfilled and worth
# backfilling with near-misses. Above it, the exact results already fill
# the typeahead and the fuzzy round trip is skipped entirely.
TITLE_SEARCH_EXACT_UNDERFILL_TRIGGER = 3

# word_similarity(query, title) cutoff for the fuzzy tier. Deliberately
# below pg_trgm's default of 0.6 so genuine typos clear the bar (a single
# typo in a multi-word title scores ~0.72, a dropped middle word ~0.87,
# messier multi-typo input ~0.45). Supplied to Postgres via a
# transaction-local set_config because the trigram GIN index only
# accelerates the `<%` operator, which reads this GUC — a bare
# `word_similarity() >= x` predicate would force a seq scan. Tunable: this
# is the one knob worth revisiting against real typeahead query logs.
TITLE_SEARCH_FUZZY_THRESHOLD = 0.45

# Minimum normalized-query length before the fuzzy tier engages. Below 3
# chars there are too few trigrams for word_similarity to be discriminating
# (one or two trigrams match a large fraction of the catalog), so short
# queries stay on the exact tiers only.
TITLE_SEARCH_FUZZY_MIN_QUERY_CHARS = 3


async def run_title_search(query: str, *, limit: int) -> TitleSearchResult:
    """Resolve a typeahead title query → ranked movie_ids.

    Args:
      query: Raw user input. Already whitespace-stripped at the API
        boundary. May still contain mixed case, diacritics, and
        punctuation — `normalize_string` collapses those to match the
        ingest-time form of `title_normalized`.
      limit: Maximum number of results to return. Caller is responsible
        for clamping into [1, TITLE_SEARCH_MAX_LIMIT].

    Returns:
      A `TitleSearchResult` whose `movie_ids` are ordered tier 1
      (token-prefix matches) first and tier 2 (substring-only matches)
      second, with the 80/20 popularity-reception prior ordering within
      each tier. When the exact tiers underfill the dropdown
      (≤ TITLE_SEARCH_EXACT_UNDERFILL_TRIGGER hits) and the query is long
      enough, a fuzzy word_similarity tier is appended below the exact
      results to absorb typos and dropped words; the total stays capped at
      `limit` and `fuzzy_count` reports how many of the ids came from that
      tier. `movie_ids` is empty (and `fuzzy_count` 0) when nothing matches
      OR when normalization collapses the query to an empty string (e.g.
      punctuation-only input like "!!!").
    """
    # Truncate first so an oversized input doesn't blow up
    # normalize_string's regex work — the regex passes are linear in
    # length, and a multi-megabyte query would be a DoS surface
    # otherwise. Truncating *before* normalization is intentional:
    # we want a bounded amount of work, not a bounded normalized
    # output (which can be longer than the input is unlikely but the
    # cap on raw input is the right guard).
    truncated = query[:TITLE_SEARCH_QUERY_MAX_CHARS]
    normalized = normalize_string(truncated)
    if not normalized:
        # All-punctuation or all-whitespace-after-normalization input.
        # Treat as "no matches" rather than an error — the typeahead
        # caller should just render an empty dropdown.
        return TitleSearchResult(movie_ids=[], fuzzy_count=0)

    # Escape LIKE metacharacters (%, _, \) so a user typing one of
    # them matches it literally, not as a wildcard. `normalize_string`
    # collapses most punctuation to space, but `_` is part of \w and
    # survives normalization — so escape_like is genuinely necessary.
    escaped = escape_like(normalized)

    # Three LIKE patterns drive the tier classification + match filter:
    #   - starts_with_pattern: title starts with the query (token at
    #     position 0).
    #   - token_prefix_pattern: query preceded by a whitespace boundary
    #     (token at position > 0). Together with starts_with_pattern
    #     these define Tier 1.
    #   - substring_pattern: query anywhere in the title — superset of
    #     the two prefix patterns. Used as the WHERE filter so a single
    #     trigram-GIN probe captures both tiers; the CASE in the SELECT
    #     classifies each row.
    starts_with_pattern = f"{escaped}%"
    token_prefix_pattern = f"% {escaped}%"
    substring_pattern = f"%{escaped}%"

    exact_ids = await fetch_title_search_movie_ids(
        starts_with_pattern=starts_with_pattern,
        token_prefix_pattern=token_prefix_pattern,
        substring_pattern=substring_pattern,
        limit=limit,
    )

    # Fuzzy fallback only when the exact tiers underfilled the dropdown AND
    # the query is long enough for trigram similarity to be meaningful. The
    # common (well-filled) typeahead path returns here with zero extra round
    # trips. `remaining` can be 0 even on the trigger path when the caller's
    # own `limit` is ≤ the underfill trigger (e.g. limit=3, exact returned
    # 3) — guard on it so we never issue a useless LIMIT 0 query.
    remaining = limit - len(exact_ids)
    if (
        len(exact_ids) <= TITLE_SEARCH_EXACT_UNDERFILL_TRIGGER
        and len(normalized) >= TITLE_SEARCH_FUZZY_MIN_QUERY_CHARS
        and remaining > 0
    ):
        # Pass the RAW normalized query, not `escaped`: the fuzzy tier uses
        # the pg_trgm `<%` operator, not LIKE, so `%`/`_` are not wildcards
        # and escaping them would corrupt the trigram tokenization.
        fuzzy_ids = await fetch_title_search_fuzzy_movie_ids(
            normalized_query=normalized,
            threshold=TITLE_SEARCH_FUZZY_THRESHOLD,
            exclude_movie_ids=set(exact_ids),
            limit=remaining,
        )
        # The fuzzy SQL already excludes the exact ids, so plain
        # concatenation is duplicate-free and keeps fuzzy strictly below
        # the exact tiers. fuzzy_count is the tail length so the API layer
        # can report how much of the dropdown came from the fuzzy tier.
        return TitleSearchResult(
            movie_ids=exact_ids + fuzzy_ids, fuzzy_count=len(fuzzy_ids)
        )

    return TitleSearchResult(movie_ids=exact_ids, fuzzy_count=0)
