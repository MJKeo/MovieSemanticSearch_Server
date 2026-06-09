# Search V2 — user query input validation.
#
# Single source of truth for sanitizing the two free-text fields a
# user can submit into the front-end pipeline: the raw `query` and the
# optional follow-up `clarification`. Every public surface that accepts
# these (the API endpoint, the streaming/non-streaming orchestrators,
# and the individually-callable Steps 0 and 1) routes through here so
# the rules — strip, non-empty, length cap — are defined once and
# enforced identically everywhere. Per the codebase convention that
# each public surface defends its own contract (it may be reached
# directly from CLI runners, notebooks, or batch scripts, not only via
# the API), these helpers are intentionally cheap and idempotent: a
# field that has already been cleaned passes through unchanged.
#
# Why a length cap at all:
#   - Cost/abuse: both fields are concatenated into the prompts sent to
#     Gemini on EVERY search. An unbounded field lets a caller inflate
#     input tokens (and therefore cost/latency) arbitrarily, and widens
#     the prompt-injection surface. A hard character ceiling bounds the
#     per-request input cost and shrinks that surface for free.
#   - It is NOT a content/abuse classifier. It only bounds size. Quota
#     abuse (request volume) and harmful-content filtering are separate
#     concerns handled elsewhere (auth + rate limiting; provider safety
#     settings) — see DIFF_CONTEXT.md.
#
# We REJECT over-length input rather than silently truncating it:
# truncation can slice a query mid-word or mid-clause and corrupt the
# user's intent, which then propagates as a confusing wrong-result set.
# A clear error lets the caller (API → 400) surface an actionable
# message instead.

from __future__ import annotations


# Maximum accepted length, in characters, of each user-supplied field.
# Both are bounded independently so a long clarification can't be used
# to bypass the query cap (the two are combined downstream). 200 is
# comfortably above a natural movie-search phrase — for reference, a
# generated spin query is capped at 150 chars (schemas/step_1.py) — and
# the clarification, a short correction, shares the same ceiling. Input
# of 200 chars passes; 201 is rejected. Tune from observed p99 query
# length if real traffic warrants it.
MAX_QUERY_CHARS = 200
MAX_CLARIFICATION_CHARS = 200


# Typed error for invalid user query input. Subclasses ValueError so the
# existing "raises ValueError on empty query" contract documented across
# the orchestrators and steps continues to hold for callers that catch
# ValueError, while giving boundaries that want to discriminate (e.g. the
# API layer mapping to HTTP 400) a precise type to catch.
class QueryInputError(ValueError):
    """Raised when a user-supplied query/clarification fails validation."""


def clean_query(raw: str) -> str:
    """Strip, validate, and return the user's raw search query.

    Args:
        raw: the user-supplied query string.

    Returns:
        The stripped query, guaranteed non-empty and within the length
        cap.

    Raises:
        QueryInputError: if the query is empty after stripping or
            exceeds ``MAX_QUERY_CHARS`` characters.
    """
    query = raw.strip()
    if not query:
        raise QueryInputError("query must be a non-empty string.")
    if len(query) > MAX_QUERY_CHARS:
        raise QueryInputError(
            f"query must be at most {MAX_QUERY_CHARS} characters "
            f"(got {len(query)})."
        )
    return query


def clean_clarification(raw: str | None) -> str | None:
    """Strip, validate, and normalize the optional clarification field.

    Empty or whitespace-only input is treated as "no clarification" so
    the pipeline's no-clarification fast path stays stable — only real
    correction text flips the steps into clarification mode.

    Args:
        raw: the user-supplied clarification string, or None.

    Returns:
        The stripped clarification, or None if it was absent/blank.

    Raises:
        QueryInputError: if the clarification exceeds
            ``MAX_CLARIFICATION_CHARS`` characters.
    """
    if not raw:
        return None
    clarification = raw.strip()
    if not clarification:
        return None
    if len(clarification) > MAX_CLARIFICATION_CHARS:
        raise QueryInputError(
            f"clarification must be at most {MAX_CLARIFICATION_CHARS} "
            f"characters (got {len(clarification)})."
        )
    return clarification
