"""
Shared scoring utilities for the movie ingestion pipeline.

Contains functions and constants used by both the Stage 3 (TMDB-only) and
Stage 5 (combined TMDB+IMDB) quality scorers.  Centralised here to avoid
duplication while keeping each stage's scorer focused on its own signal
definitions and weights.
"""

import datetime
import math
import struct
from enum import StrEnum

# ---------------------------------------------------------------------------
# Age-adjustment constants — shared by Stage 3 and Stage 5 vote_count scoring
# ---------------------------------------------------------------------------

# Days after release before absence of US streaming providers is penalised.
THEATER_WINDOW_DAYS: int = 75

# Recency boost: films < 2 years old get a proportional multiplier up to
# VC_RECENCY_BOOST_MAX×, compensating for shorter vote-accumulation windows.
# At exactly 1 year the multiplier is 2.0×; it decays hyperbolically to 1.0×
# at 2 years, then disappears.
VC_RECENCY_BOOST_MAX: float = 2.0

# Classic boost: films older than VC_CLASSIC_START_YEARS receive a linearly
# growing multiplier capped at VC_CLASSIC_BOOST_CAP, compensating for
# pre-internet-era underrepresentation in online voting platforms.
VC_CLASSIC_START_YEARS: int = 20   # age at which the boost begins (years)
VC_CLASSIC_RAMP_YEARS: int = 30    # years of linear ramp from 1× to cap
VC_CLASSIC_BOOST_CAP: float = 1.5  # maximum classic-film multiplier

# Popularity log-scale cap: just above p99 (8.95).  Used by both stages.
POP_LOG_CAP: int = 11


# ---------------------------------------------------------------------------
# Vote count source — determines the log cap for vote_count scoring
# ---------------------------------------------------------------------------


class VoteCountSource(StrEnum):
    """Selects the appropriate log cap for vote_count scoring.

    TMDB votes are concentrated in a lower range (p99 ≈ 1821), so the cap is
    2001.  IMDB votes are higher (p90 ≈ 10,625), so the cap is 12001 to
    spread discrimination across the 100–10,000 range where borderline movies
    live.
    """
    TMDB = "tmdb"
    IMDB = "imdb"


# Log caps keyed by source.  Each cap is set just above the relevant
# percentile ceiling so log10(vc+1)/log10(cap) saturates at 1.0 for movies
# that are unambiguously notable.
_VC_LOG_CAPS: dict[VoteCountSource, int] = {
    VoteCountSource.TMDB: 2001,
    VoteCountSource.IMDB: 12001,
}


# ---------------------------------------------------------------------------
# BLOB decoding
# ---------------------------------------------------------------------------


def unpack_provider_keys(blob: bytes | None) -> list[int]:
    """Unpack a watch_provider_keys BLOB into a list of integer provider keys.

    The BLOB is packed as little-endian unsigned 32-bit integers ('<NI' format),
    matching the encoding written by tmdb_fetcher.py.  Returns an empty list for
    None or zero-length input.
    """
    if not blob:
        return []
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}I", blob[:count * 4]))


# ---------------------------------------------------------------------------
# Shared signal scoring functions
# ---------------------------------------------------------------------------


def score_vote_count(
    vc: int,
    release_date: str | None,
    today: datetime.date,
    source: VoteCountSource,
) -> float:
    """Log-scaled vote_count score in [0, 1] with age-based multipliers.

    The log scale compresses the long tail: the gap between 1 and 10 votes
    matters far more than the gap between 5000 and 10000.  The cap is
    source-dependent (TMDB uses 2001, IMDB uses 12001) because the two
    platforms have different vote-count distributions.

    Two non-overlapping age multipliers correct for systematic bias:

      Recency multiplier (films < 2 years old):
        Recent films have had less time to accumulate votes.  The multiplier
        is VC_RECENCY_BOOST_MAX (2.0×) for films ≤ 1 year old and decays
        hyperbolically to 1.0× at 2 years, beyond which it is floored at
        1.0 so no penalty is applied to middle-aged or old films.

      Classic multiplier (films > VC_CLASSIC_START_YEARS years old):
        Films made before online voting platforms were widely adopted are
        chronically under-rated relative to their true cultural significance.
        The multiplier grows linearly from 1.0× at 20yr to 1.5× at 50yr.

    The two windows are mutually exclusive (0–2yr vs >20yr); the larger of
    the two multipliers is applied.  Films aged 2–20 years receive no
    adjustment (multiplier = 1.0×).
    """
    log_cap = _VC_LOG_CAPS[source]
    base = min(math.log10(vc + 1) / math.log10(log_cap), 1.0)

    if release_date is not None:
        try:
            release = datetime.date.fromisoformat(release_date)
            # Floor at 0.5yr to avoid division instability for very new films.
            age_years = max((today - release).days / 365.0, 0.5)

            # Recency: 2× at ≤1yr, hyperbolic decay to 1× at 2yr.
            recent_boost = max(
                1.0,
                min(VC_RECENCY_BOOST_MAX, VC_RECENCY_BOOST_MAX / age_years),
            )

            # Classic: linear ramp from 1× at 20yr to 1.5× at 50yr.
            classic_boost = min(
                VC_CLASSIC_BOOST_CAP,
                1.0 + max(0.0, age_years - VC_CLASSIC_START_YEARS) / VC_CLASSIC_RAMP_YEARS,
            )

            # Apply the larger of the two adjustments.
            multiplier = max(recent_boost, classic_boost)
            base = min(base * multiplier, 1.0)
        except ValueError:
            # Non-parseable date — use base score unchanged.
            pass

    return base


def score_popularity(popularity: float) -> float:
    """Log-scaled popularity score in [0, 1], capped at popularity=10.

    TMDB's algorithmic activity score (page views, watchlist additions, etc.).
    Complementary to vote_count — captures current momentum for new releases
    and cult titles.  The cap at 10 places the ceiling just above p99 (8.95).
    """
    return min(math.log10(max(popularity, 0.0) + 1) / math.log10(POP_LOG_CAP), 1.0)


def validate_weights(weights: dict[str, float], label: str = "WEIGHTS") -> None:
    """Raise ValueError if weights don't sum to 1.0.

    Called at module load time by both Stage 3 and Stage 5 scorers to catch
    weight table corruption before any scoring runs.  Uses an explicit
    ValueError rather than assert because assert is silently suppressed
    under ``python -O``.
    """
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"{label} must sum to 1.0; got {total:.10f}"
        )
