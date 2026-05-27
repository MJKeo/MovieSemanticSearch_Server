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
# each tier, ranking uses the neutral 80/20 popularity-reception
# blend; ties break by `movie_id DESC` for stability.
#
# Fuzzy matching (edit-distance ≤ 2 on queries ≥ 4 chars) was marked
# optional in the spec and is deliberately omitted from v1 — it would
# require either a second SQL pass with `levenshtein()` or extending
# the SELECT with a similarity threshold, both of which trade against
# the p50<20ms latency target. Easy to add later if the UX warrants.
#
# Wire-level validation (`q` non-empty after trim, `limit` in [1, 25])
# lives in api/main.py — this module sees the already-validated query.

from __future__ import annotations

from db.postgres import fetch_title_search_movie_ids
from implementation.misc.helpers import normalize_string
from implementation.misc.sql_like import escape_like


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


async def run_title_search(query: str, *, limit: int) -> list[int]:
    """Resolve a typeahead title query → ranked movie_ids.

    Args:
      query: Raw user input. Already whitespace-stripped at the API
        boundary. May still contain mixed case, diacritics, and
        punctuation — `normalize_string` collapses those to match the
        ingest-time form of `title_normalized`.
      limit: Maximum number of results to return. Caller is responsible
        for clamping into [1, TITLE_SEARCH_MAX_LIMIT].

    Returns:
      Ordered list of `movie_id`s, tier 1 (token-prefix matches) first
      and tier 2 (substring-only matches) second, with the 80/20
      popularity-reception prior ordering within each tier. Empty list
      when nothing matches OR when normalization collapses the query
      to an empty string (e.g. punctuation-only input like "!!!").
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
        return []

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

    return await fetch_title_search_movie_ids(
        starts_with_pattern=starts_with_pattern,
        token_prefix_pattern=token_prefix_pattern,
        substring_pattern=substring_pattern,
        limit=limit,
    )
