# Search V2 — Stage 4 system prior scoring.
#
# Per-candidate quality / notability prior scores plus the composite
# used to seed the exclusion-only browse pool. Formulas mirror the
# metadata endpoint's well-received / poorly-received and popular /
# niche ramps so standard and inverted priors share a single scoring
# shape (see metadata_query_execution.py::_score_reception and
# _score_popularity).
#
# Inputs are whatever movie_card serves:
#   reception_score ∈ [0, 100] or None
#   popularity_score ∈ [0, 1]  or None
# Every helper returns a value in [0, 1]; None-inputs collapse to 0.
#
# Callers:
#   scoring.py uses quality_score / notability_score to contribute a
#   (weight, score) pair into the preference weighted average.
#   orchestrator.py uses browse_composite to order the BROWSE flow's
#   movie_card seed pool.

from __future__ import annotations

from schemas.enums import SystemPrior


# Reception ramps follow the metadata endpoint exactly:
#   high:  (reception - 55) / 40, clamp [0,1]  (55→0, 95→1)
#   low :  (50 - reception) / 40, clamp [0,1]  (50→0, 10→1)
_RECEPTION_HIGH_FLOOR = 55.0
_RECEPTION_LOW_CEIL = 50.0
_RECEPTION_SPAN = 40.0

# Relative weights inside the quality prior composite. Reception is
# the primary quality signal; popularity is a weaker corroborator for
# when reception data is sparse. Documented as tunable.
_QUALITY_RECEPTION_WEIGHT = 0.7
_QUALITY_POPULARITY_WEIGHT = 0.3

# Browse seed combines quality and notability evenly — no strong
# reason to prefer one over the other when there is zero explicit
# intent from the user.
_BROWSE_QUALITY_WEIGHT = 0.5
_BROWSE_NOTABILITY_WEIGHT = 0.5


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _reception_high(reception: float | None) -> float:
    if reception is None:
        return 0.0
    return _clamp((reception - _RECEPTION_HIGH_FLOOR) / _RECEPTION_SPAN)


def _reception_low(reception: float | None) -> float:
    if reception is None:
        return 0.0
    return _clamp((_RECEPTION_LOW_CEIL - reception) / _RECEPTION_SPAN)


def _pop_high(popularity: float | None) -> float:
    if popularity is None:
        return 0.0
    return _clamp(popularity)


def _pop_low(popularity: float | None) -> float:
    if popularity is None:
        return 0.0
    return _clamp(1.0 - popularity)


def quality_score(
    reception: float | None,
    popularity: float | None,
    mode: SystemPrior,
) -> float:
    """Per-candidate quality prior score in [0, 1].

    ENHANCED and STANDARD share the same score shape (weight differs,
    not score). INVERTED mirrors it across the "bad" ramps. SUPPRESSED
    returns 0 — callers should omit the prior from the weighted sum
    rather than feed 0 in with a 0 weight (either path yields the
    same numerator but keeps the denominator cleaner).
    """
    if mode in (SystemPrior.ENHANCED, SystemPrior.STANDARD):
        return _clamp(
            _QUALITY_RECEPTION_WEIGHT * _reception_high(reception)
            + _QUALITY_POPULARITY_WEIGHT * _pop_high(popularity)
        )
    if mode == SystemPrior.INVERTED:
        return _clamp(
            _QUALITY_RECEPTION_WEIGHT * _reception_low(reception)
            + _QUALITY_POPULARITY_WEIGHT * _pop_low(popularity)
        )
    return 0.0


def notability_score(
    popularity: float | None,
    mode: SystemPrior,
) -> float:
    """Per-candidate notability prior score in [0, 1].

    Notability tracks only popularity_score by design: reception is
    about quality judgment, not about how widely known a movie is.
    """
    if mode in (SystemPrior.ENHANCED, SystemPrior.STANDARD):
        return _pop_high(popularity)
    if mode == SystemPrior.INVERTED:
        return _pop_low(popularity)
    return 0.0


def browse_composite(
    reception: float | None,
    popularity: float | None,
    quality_mode: SystemPrior,
    notability_mode: SystemPrior,
) -> float:
    """Composite score used to order the BROWSE seed pool.

    Same underlying ramps as quality_score / notability_score — single
    source of truth. When a prior is SUPPRESSED for scoring purposes,
    the browse seed still needs to return *something* ordered, so we
    fall back to STANDARD mode inside browse ordering only. Scoring
    callers drop suppressed priors from the weighted average entirely.

    Mirror: the same arithmetic is expressed in SQL inside
    ``db.postgres.fetch_browse_seed_ids`` so Postgres can do the
    scoring / sorting / LIMIT without materializing the full corpus.
    Keep the two implementations in sync when formulas change.
    """
    q_effective_mode = (
        SystemPrior.STANDARD
        if quality_mode == SystemPrior.SUPPRESSED
        else quality_mode
    )
    n_effective_mode = (
        SystemPrior.STANDARD
        if notability_mode == SystemPrior.SUPPRESSED
        else notability_mode
    )
    q = quality_score(reception, popularity, q_effective_mode)
    n = notability_score(popularity, n_effective_mode)
    return _clamp(
        _BROWSE_QUALITY_WEIGHT * q + _BROWSE_NOTABILITY_WEIGHT * n
    )
