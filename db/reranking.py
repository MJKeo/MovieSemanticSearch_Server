"""
reranking.py — Quality prior reranking for search candidates.

Candidates are bucketed by relevance (rounded final_score) and sorted
within each bucket by a normalized reception score. This prevents
low-quality movies from ranking above high-quality ones when their
relevance scores are essentially tied, while still respecting explicit
"poorly received" user preferences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from implementation.classes.enums import ReceptionType

if TYPE_CHECKING:
    from db.search import SearchCandidate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reception score range used for min-max normalization.
# Scores below FLOOR are treated as 0.0; scores above CEIL are treated as 1.0.
RECEPTION_FLOOR = 30.0
RECEPTION_CEIL = 90.0

# Number of decimal places used when bucketing final_score for tie-breaking.
BUCKET_PRECISION = 2


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_reception(raw: float | None) -> float:
    """
    Map a raw reception score into [0, 1] via min-max normalization.

    Clamps to [RECEPTION_FLOOR, RECEPTION_CEIL] before scaling.
    Returns 0.5 (neutral midpoint) when the raw score is None/missing.
    """
    if raw is None:
        return 0.5

    # Clamp to the defined range, then scale to [0, 1]
    clamped = max(RECEPTION_FLOOR, min(RECEPTION_CEIL, raw))
    return (clamped - RECEPTION_FLOOR) / (RECEPTION_CEIL - RECEPTION_FLOOR)


def compute_quality_prior(
    reception_score: float | None,
    reception_type: ReceptionType,
) -> float:
    """
    Compute the quality prior used for within-bucket tie-breaking.

    When the user explicitly asks for poorly-received movies, the quality
    prior is zeroed out so that reception doesn't interfere with the
    relevance ranking. Otherwise, the prior is the normalized reception score.
    """
    if reception_type == ReceptionType.POORLY_RECEIVED:
        return 0.0

    return normalize_reception(reception_score)


# ---------------------------------------------------------------------------
# Reranking entry point
# ---------------------------------------------------------------------------

def rerank_candidates(
    candidates: list[SearchCandidate],
    reception_scores: dict[int, float | None],
    reception_type: ReceptionType,
) -> None:
    """
    Rerank candidates in-place by bucketed relevance, then quality prior.

    Steps:
        1. Round each candidate's final_score to BUCKET_PRECISION decimals.
        2. Compute a quality_prior from the movie's reception score.
        3. Store both values on the candidate.
        4. Sort by (-bucketed_final_score, -quality_prior) so that within
           the same relevance bucket, higher-quality movies surface first.

    Args:
        candidates: Merged search candidates with final_score already set.
        reception_scores: Mapping of movie_id -> raw reception score (may be None).
        reception_type: The user's reception preference from query understanding.
    """
    for c in candidates:
        c.bucketed_final_score = round(c.final_score, BUCKET_PRECISION)
        c.quality_prior = compute_quality_prior(
            reception_scores.get(c.movie_id),
            reception_type,
        )

    candidates.sort(key=lambda c: (-c.bucketed_final_score, -c.quality_prior))
