"""Franchise-specific text normalization and tokenization.

Mirrors `production_company_text.py` for the franchise resolution path (see
search_improvement_planning/v2_search_data_improvements.md, "Franchise
Resolution"). Applies the shared `normalize_string` rules and the shared
digit-word rewrites (ordinal `20th` → `twentieth`, cardinal 0–99 `1` →
`one`), then tokenizes on whitespace and hyphens, and finally drops the
closed-form `FRANCHISE_STOPLIST` below.

Applied symmetrically at ingest and query time so variants like
`phase 1` / `phase one`, `the lord of the rings` / `lord of the rings`,
and `spider-man` / `spider man` collide to the same token set.

Only applies to franchise strings (lineage, shared_universe, recognized
subgroups). We deliberately share the digit-word substitution helper with
production-company normalization rather than duplicate the 0–99 word table
— one source of truth prevents drift between the two resolvers.
"""

from __future__ import annotations

from implementation.misc.helpers import normalize_string, tokenize_title_phrase
from implementation.misc.production_company_text import apply_digit_word_substitution


# Closed stopword droplist, applied symmetrically at ingest and query
# time. Source: v2_search_data_improvements.md §Franchise Resolution /
# Stage C. These are pure English function words whose presence or
# absence carries no franchise signal — `"the lord of the rings"` and
# `"lord of the rings"` must collide, and `"a" in "a clockwork orange"`
# must not anchor a posting-list intersection.
#
# Scaffolding tokens (`trilogy`, `collection`, `films`, `series`,
# `universe`, `cinematic`, `chronicles`, `anthology`, `franchise`) are
# deliberately KEPT — their DF range overlaps with load-bearing
# discriminative tokens like `man` (Spider-Man / Iron Man / Batman /
# Ant-Man) and `dead` (Evil Dead / Dead Poets), so any statistical
# cutoff wide enough to remove scaffolding also removes signal. See
# the "Why Not a DF Ceiling" subsection in the planning doc for the
# full rationale.
#
# `lex.franchise_token_doc_frequency` exists as a diagnostic for
# curating new stopword candidates from real data over time; additions
# should be made by hand here rather than by numeric threshold.
FRANCHISE_STOPLIST: frozenset[str] = frozenset({
    "the", "of", "and", "a", "in", "to", "on",
    "my", "i", "for", "at", "by", "with",
})


def normalize_franchise_string(raw: str) -> str:
    """Normalize a franchise lineage / shared_universe / subgroup string.

    Pipeline mirrors `normalize_company_string`:
      1. Shared `normalize_string` (NFC → casefold → NFKD diacritic fold →
         punctuation→space except `-` → collapse whitespace → strip).
      2. Ordinal number-to-word (`phase 1st` → `phase first`).
      3. Cardinal number-to-word for pure-numeric tokens in [0, 99]
         (`phase 1` → `phase one`, `fast 2 furious` → `fast two furious`).

    Returns an empty string when the input normalizes to empty
    (punctuation-only input, None, etc.) — callers skip empty normalized
    strings before upserting into `lex.franchise_entry`.
    """
    base = normalize_string(raw)
    if not base:
        return ""
    return apply_digit_word_substitution(base)


def tokenize_franchise_string(
    raw: str, *, already_normalized: bool = False
) -> list[str]:
    """Tokenize a franchise string for `lex.franchise_token` insertion.

    Delegates to the shared title tokenizer (whitespace split + hyphen
    expansion + dedup), then drops `FRANCHISE_STOPLIST` entries and any
    lone-hyphen residue. The `already_normalized` hand-off skips a redundant
    `normalize_franchise_string` pass when the caller already has the
    normalized form — used in the per-movie ingest hot loop where the same
    normalized string feeds both the `franchise_entry` upsert and the token
    set.

    Stopword drop is applied symmetrically at ingest and query time so
    `"the lord of the rings"` and `"lord of the rings"` produce the same
    token set `{lord, rings}`. Mirrors the award-name resolver's
    `AWARD_STOPLIST` handling in `award_name_text.py`.

    Drops lone-hyphen tokens for the same reason `tokenize_company_string`
    does: `normalize_string` preserves `-` as a character, which can leave a
    standalone `-` token after whitespace-splitting on strings shaped like
    `X - Y`. A bare hyphen carries no matching signal.
    """
    normalized = raw if already_normalized else normalize_franchise_string(raw)
    if not normalized:
        return []
    tokens = tokenize_title_phrase(normalized, already_normalized=True)
    return [
        t for t in tokens
        if t.replace("-", "") and t not in FRANCHISE_STOPLIST
    ]
