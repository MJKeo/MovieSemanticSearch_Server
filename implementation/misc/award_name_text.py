"""Award-name-specific text normalization and tokenization.

Mirrors the shape of production_company_text.py / franchise_text.py so the
three freeform resolvers share one mental model. The base
``normalize_string`` lowercases, folds diacritics (NFKD → strip combining
marks), strips punctuation (curly and straight apostrophes collapse to
nothing), and collapses whitespace — straight-vs-curly apostrophe splits
like ``Palme d'Or`` / ``Palme d\u2019Or`` and ``Critics Week`` /
``Critics' Week`` merge to a single normalized key. On top of that we
apply the ordinal + cardinal digit-to-word rewrite already shared by
studio and franchise so ``8th Annual Critics' Week Grand Prize`` stays
token-compatible with any future natural-language phrasings.

Ingest and query share ``normalize_award_string`` verbatim, so the entry
lookup key is identical on both sides — that symmetry is what replaces
the v1 exact-equality comparison. The tokenizers intentionally diverge
by one step: ``tokenize_award_string`` (ingest) emits every surviving
token, while ``tokenize_award_string_for_query`` (query) additionally
drops ``AWARD_QUERY_STOPLIST`` before returning. The ingest side
deliberately keeps every token in ``lex.award_name_token`` so the
droplist can be revised from the DF materialized view without
re-ingesting. The finalized droplist and the reasoning against a
numeric DF ceiling live in
search_improvement_planning/v2_search_data_improvements.md
§ "Stopword Droplist" / "Why Not a DF Ceiling".
"""

from __future__ import annotations

from implementation.misc.helpers import normalize_string, tokenize_title_phrase
from implementation.misc.production_company_text import apply_digit_word_substitution


def normalize_award_string(raw: str) -> str:
    """Normalize an IMDB ``award_name`` for lex.award_name_entry lookup.

    Runs the shared ``normalize_string`` (lowercase, diacritic fold,
    punctuation strip, whitespace collapse; hyphens preserved) then
    ``apply_digit_word_substitution`` so ordinals (``8th`` → ``eighth``)
    and small cardinals in [0, 99] (``8`` → ``eight``, ``01`` → ``one``)
    collide with their word forms. Returns the empty string when the
    input normalizes to nothing so callers can skip it cleanly.

    Pre-folds curly/"smart" single quotation marks (U+2018, U+2019) to
    the ASCII apostrophe before normalization. ``normalize_string``'s
    apostrophe-strip regex only covers ASCII and a handful of modifier
    variants — not U+2018 / U+2019 — which is why IMDB's
    ``Palme d'Or`` (U+2019) and a user-typed ``Palme d'Or`` (U+0027)
    would otherwise normalize to different strings (``palme d or`` vs
    ``palme dor``). Fixing the shared helper would invalidate every
    existing lex.lexical_dictionary / lex.production_company /
    lex.franchise_entry row that was stamped under the old rule, so the
    fold is scoped here and mirrored symmetrically at query time.
    """
    raw = raw.replace("\u2018", "'").replace("\u2019", "'")
    base = normalize_string(raw)
    if not base:
        return ""
    return apply_digit_word_substitution(base)


def tokenize_award_string(raw: str, *, already_normalized: bool = False) -> list[str]:
    """Tokenize an award name for the ingest-side posting path.

    Delegates to ``tokenize_title_phrase`` (whitespace + hyphen split,
    dedup) and emits every surviving token. No stoplist is applied —
    ingest keeps every token in ``lex.award_name_token`` so the
    query-side droplist (``AWARD_QUERY_STOPLIST``, applied via
    ``tokenize_award_string_for_query``) can be revised without
    re-ingest. Only lone-hyphen residue (from names shaped like
    ``X - Y Award``, where ``normalize_string`` preserves the bare
    ``-`` as a standalone token) is filtered, matching the guard in
    ``tokenize_company_string``.

    Pass ``already_normalized=True`` in the hot ingest loop where the
    caller has already run ``normalize_award_string`` on the input —
    skips the redundant normalize + digit-substitute pair on each token.
    """
    normalized = raw if already_normalized else normalize_award_string(raw)
    if not normalized:
        return []
    tokens = tokenize_title_phrase(normalized, already_normalized=True)
    return [t for t in tokens if t.replace("-", "")]


# Finalized query-side droplist. See
# search_improvement_planning/v2_search_data_improvements.md §"Stopword
# Droplist" for the empirical top-DF distribution that drove the choice
# and the reasoning against a numeric DF ceiling. This asymmetry vs
# franchise (which applies its stoplist bilaterally) is deliberate —
# the ingest path writes every token to lex.award_name_token so the
# list can be revised from the DF materialized view without re-ingest.
AWARD_QUERY_STOPLIST: frozenset[str] = frozenset(
    {
        "award",
        "awards",
        "prize",
        "prizes",
        "film",
        "films",
        "best",
        "a",
        "an",
        "and",
        "for",
        "of",
        "the",
    }
)


def tokenize_award_string_for_query(raw: str) -> list[str]:
    """Tokenize an award name for the query-side posting-list lookup.

    Runs the same normalize + whitespace/hyphen split as the ingest
    tokenizer (``tokenize_award_string``), then drops
    ``AWARD_QUERY_STOPLIST`` tokens before they reach
    ``lex.award_name_token``. Returning an empty list signals to the
    caller that this name contributes nothing to the axis (every token
    was a stopword), in which case the executor should skip it rather
    than broaden the posting-list fetch.
    """
    tokens = tokenize_award_string(raw)
    return [t for t in tokens if t not in AWARD_QUERY_STOPLIST]
