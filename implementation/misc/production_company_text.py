"""Production-company-specific text normalization and tokenization.

Wraps the shared `normalize_string` and `tokenize_title_phrase` helpers with
the "ordinal number-to-word" rule called for by the v2 studio spec (D2 in
search_improvement_planning/v2_search_data_improvements.md). The rule rewrites
tokens like `20th` → `twentieth` and `21st` → `twenty-first` so the numeric
and worded variants of `20th Century Fox` / `Twentieth Century Fox` collide
to the same normalized key and the same token set.

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


def normalize_company_string(raw: str) -> str:
    """Normalize an IMDB `production_companies` string for dictionary lookup.

    Applies the shared `normalize_string` rules (lowercase, diacritic fold,
    punctuation strip, whitespace collapse; preserves hyphens) and then
    rewrites every ordinal like `20th` to its word form (`twentieth`). This
    ensures `20th Century Fox` and `Twentieth Century Fox` normalize to the
    same key and therefore collapse into a single `lex.production_company`
    row.
    """
    base = normalize_string(raw)
    if not base:
        return ""
    return _ORDINAL_RE.sub(_ordinal_sub, base)


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
    return tokenize_title_phrase(normalized, already_normalized=True)
