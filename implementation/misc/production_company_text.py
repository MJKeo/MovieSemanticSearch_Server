"""Production-company-specific text normalization and tokenization.

Wraps the shared `normalize_string` and `tokenize_title_phrase` helpers with
two digit-to-word rules called for by the v2 studio spec (D2 in
search_improvement_planning/v2_search_data_improvements.md):

- **Ordinal** number-to-word (`20th` → `twentieth`, `21st` → `twenty-first`)
  so `20th Century Fox` / `Twentieth Century Fox` collide to the same key.
- **Cardinal** number-to-word for pure-numeric tokens in [0, 99]
  (`Section 8 Productions` → `section eight productions`, `Studio 01` →
  `studio one`) so small numbers that users plausibly type either form match
  either way. Numbers ≥ 100 stay in digit form since year-like numbers
  (`Fox 2000`, `Studio 100`) are never spelled out by users.

Also drops lone-hyphen tokens (from names shaped like `X - Y Productions`)
during tokenization — a bare `-` carries no matching signal.

Applied at both ingest time (when stamping `lex.production_company` rows) and
query time (when resolving the freeform studio path), preserving the
normalizer's "both sides see the same string" invariant.

Only applies to production-company strings. The shared `normalize_string` in
helpers.py is intentionally left alone — rewriting it globally would
invalidate every existing lex.lexical_dictionary row (titles, actors, etc.).
"""

from __future__ import annotations

import re

from implementation.misc.helpers import normalize_string, tokenize_title_phrase


# Ordinal word forms for 1–30. Covers every ordinal that plausibly appears in
# IMDB production_companies strings (`20th Century Fox`, `21st Century Fox`,
# `1st Look Pictures`, etc.). Beyond 30 the long tail is too sparse to matter.
ORDINAL_WORDS: dict[int, str] = {
    1: "first",         2: "second",        3: "third",         4: "fourth",
    5: "fifth",         6: "sixth",         7: "seventh",       8: "eighth",
    9: "ninth",         10: "tenth",        11: "eleventh",     12: "twelfth",
    13: "thirteenth",   14: "fourteenth",   15: "fifteenth",    16: "sixteenth",
    17: "seventeenth",  18: "eighteenth",   19: "nineteenth",   20: "twentieth",
    21: "twenty-first", 22: "twenty-second", 23: "twenty-third",
    24: "twenty-fourth", 25: "twenty-fifth", 26: "twenty-sixth",
    27: "twenty-seventh", 28: "twenty-eighth", 29: "twenty-ninth",
    30: "thirtieth",
}

# Matches a bare ordinal like `20th`, `21st`, `1st`, `2nd`, `3rd`, `4th`.
# Word boundaries prevent mid-token substitution (e.g., `film20th` stays
# intact). After normalize_string the input is already lowercased, so only
# lowercase suffixes need to be matched.
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b")


def _ordinal_sub(match: re.Match[str]) -> str:
    """Regex replacement: map `\\d+(st|nd|rd|th)` to its word form.

    Leaves numbers outside ORDINAL_WORDS untouched so `100th Anniversary
    Productions` doesn't get corrupted into something half-translated.
    """
    number = int(match.group(1))
    word = ORDINAL_WORDS.get(number)
    return word if word is not None else match.group(0)


# Cardinal word forms for 0–19. Tens (20, 30, ...) and their compounds
# (21 = `twenty-one`, etc.) are derived from `_CARDINAL_TENS`. We cap at
# 99 because: (a) users rarely type three-digit numbers as words, and
# (b) year-like numbers (`1984`, `2000`) embedded in company names should
# stay in digit form — converting `Fox 2000` to `Fox two thousand` adds
# unnatural tokens that query-side users would never produce.
_CARDINAL_ONES: dict[int, str] = {
    0: "zero",       1: "one",        2: "two",        3: "three",
    4: "four",       5: "five",       6: "six",        7: "seven",
    8: "eight",      9: "nine",       10: "ten",       11: "eleven",
    12: "twelve",    13: "thirteen",  14: "fourteen",  15: "fifteen",
    16: "sixteen",   17: "seventeen", 18: "eighteen",  19: "nineteen",
}
_CARDINAL_TENS: dict[int, str] = {
    2: "twenty", 3: "thirty",  4: "forty",  5: "fifty",
    6: "sixty",  7: "seventy", 8: "eighty", 9: "ninety",
}

# Match any purely-numeric run bounded by non-word characters. Leading
# zeros (`01`) are captured by `\d+` and collapsed to the integer value
# before lookup, so `Studio 01` and `Studio 1` both resolve to `one`.
# Cardinal substitution runs AFTER ordinal substitution, so `20th` has
# already become `twentieth` by the time this regex runs and will not
# match its leftover digits here.
_CARDINAL_RE = re.compile(r"\b(\d+)\b")


def _cardinal_to_word(n: int) -> str | None:
    """Return the English word form for cardinal `n` in [0, 99], else None.

    Compounds use hyphens (`twenty-one`), matching the ordinal dict's
    convention. The tokenizer later splits on hyphens, so `twenty-one`
    contributes `{twenty-one, twenty, one}` to the token set — maximizing
    matching against both hyphenated and space-separated query variants.
    """
    if n < 0 or n > 99:
        return None
    if n < 20:
        return _CARDINAL_ONES[n]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _CARDINAL_TENS[tens]
    return f"{_CARDINAL_TENS[tens]}-{_CARDINAL_ONES[ones]}"


def _cardinal_sub(match: re.Match[str]) -> str:
    """Regex replacement: map a pure-numeric token to its word form when in
    range, else leave it unchanged so large numbers (years, quantities) keep
    their digit form."""
    number = int(match.group(1))
    word = _cardinal_to_word(number)
    return word if word is not None else match.group(0)


def normalize_company_string(raw: str) -> str:
    """Normalize an IMDB `production_companies` string for dictionary lookup.

    Applies the shared `normalize_string` rules (lowercase, diacritic fold,
    punctuation strip, whitespace collapse; preserves hyphens), then
    rewrites ordinals (`20th` → `twentieth`) and pure-numeric cardinals in
    [0, 99] (`8` → `eight`, `01` → `one`, `20` → `twenty`). Together these
    collapse digit/word variants like `20th Century Fox` / `Twentieth Century
    Fox` and `Section 8 Productions` / `Section Eight Productions` into the
    same normalized key (and the same token set for the freeform path).

    Ordinal substitution runs before cardinal substitution so the digit
    portion of `20th` is consumed by the ordinal rule and never seen by the
    cardinal rule.
    """
    base = normalize_string(raw)
    if not base:
        return ""
    base = _ORDINAL_RE.sub(_ordinal_sub, base)
    base = _CARDINAL_RE.sub(_cardinal_sub, base)
    return base


def tokenize_company_string(raw: str, *, already_normalized: bool = False) -> list[str]:
    """Tokenize a production-company string for the freeform studio path.

    Delegates to the shared title tokenizer's split logic (whitespace split
    + hyphen expansion, dedup, no stoplist). The inner `already_normalized`
    hand-off avoids re-running `normalize_string` from inside the tokenizer.

    Args:
        raw: A production-company string.
        already_normalized: Pass True when the caller has already run
            ``normalize_company_string`` on the input. Skips the redundant
            normalize + ordinal-substitute pair that would otherwise run
            per token — meaningful when this is called in a hot loop like
            the per-movie ingest fan-out, where the same normalized string
            is used for both the dictionary upsert and tokenization.
    """
    normalized = raw if already_normalized else normalize_company_string(raw)
    if not normalized:
        return []
    tokens = tokenize_title_phrase(normalized, already_normalized=True)
    # Guard against lone-hyphen tokens from names like ``X - Y Productions``,
    # where normalize_string preserves ``-`` as a character and the ``- ``
    # between words survives whitespace-splitting as a standalone ``-`` token.
    # A bare hyphen carries no matching signal; dropping it keeps studio_token
    # clean without altering the shared title tokenizer's behavior.
    return [t for t in tokens if t.replace("-", "")]
