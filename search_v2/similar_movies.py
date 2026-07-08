"""Standalone similar-movies search flow.

This module owns the Step-0 similarity path and the tmdb_id-facing debug
entrypoint. It is intentionally separate from the standard Stage-4 search
pipeline: no standard trait decomposition, endpoint execution, or branch
reranking code is called here.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Sized
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, TypeVar

import numpy as np
from opentelemetry import trace

from db.postgres import (
    SimilarityAwardSignals,
    TRAIT_KIND_CONCEPT_TAG,
    TRAIT_KIND_OVERALL_KEYWORD,
    TRAIT_KIND_SOURCE_MATERIAL,
    TRAIT_KIND_TMDB_GENRE,
    fetch_director_movie_terms,
    fetch_director_term_ids_for_movies,
    fetch_movie_ids_by_overall_keywords,
    fetch_movie_ids_by_themes_recall,
    fetch_movie_ids_by_production_company_ids,
    fetch_movie_ids_with_title_like,
    fetch_production_company_ids_by_normalized_strings,
    fetch_similarity_award_category_tags,
    fetch_similarity_award_signals,
    fetch_similarity_franchise_candidates,
    fetch_similarity_quality_candidates,
    fetch_similarity_signal_rows,
    fetch_similarity_source_candidates,
    fetch_similarity_top_billed_cast,
    fetch_trait_idfs,
)
from search_v2.auteur_directors import fetch_auteur_term_ids
from db.qdrant import qdrant_client
from qdrant_client.http.models import Filter, QueryRequest
from db.vector_scoring import normalize_blended_scores
from db.vector_search import COLLECTION_ALIAS, QDRANT_SEARCH_PARAMS, build_qdrant_filter
from implementation.classes.enums import VectorName
from implementation.classes.overall_keywords import OverallKeyword
from implementation.classes.schemas import MetadataFilters
from implementation.misc.helpers import normalize_string
from implementation.misc.sql_like import escape_like
from schemas.enums import AwardCeremony
from schemas.step_0_flow_routing import SimilarityFlowData, SimilarityReference
from search_v2.award_taxonomy import (
    SPECIFICITY_FACTOR,
    TIER_WEIGHT,
    tag_level,
)
from search_v2.country_language_registry import (
    COUNTRY_LANGUAGE_KEYWORD_IDS,
    US_DEFAULT,
    country_set,
)
from search_v2.format_registry import (
    FORMAT_KEYWORD_IDS_ALL,
    FormatBucket,
    format_bucket,
)
from search_v2.production_medium_registry import (
    MEDIUM_TAG_IDS,
    load_medium_idfs,
    medium_score,
)
from search_v2.similar_studio_registry import (
    StudioSimilarityEntry,
    studio_entries_by_normalized_string,
)

from observability.names import (
    # query_search-branch-specific: the reference->anchor resolution skeleton,
    # set only on the /query_search path (the pure endpoint supplies anchor IDs).
    QUERY_SEARCH_BRANCH_ENTITIES,
    QUERY_SEARCH_BRANCH_ENTITY_RESOLVED_COUNTS,
    QUERY_SEARCH_BRANCH_UNRESOLVED_ENTITY_COUNT,
    # flow-neutral engine spans + signal attributes (land on the branch span on the
    # /query_search path, on the server span on the /similarity_search path).
    SIMILARITY_ADDITIONAL_BOOSTS,
    SIMILARITY_ANCHOR_COUNT,
    SIMILARITY_ANCHOR_SHAPE,
    SIMILARITY_ANCHOR_SHAPE_COHESION,
    SIMILARITY_FETCH,
    SIMILARITY_FETCH_LANE,
    SIMILARITY_FETCH_MATCH,
    SIMILARITY_FETCH_RESULT_COUNT,
    SIMILARITY_LANE_COHESION,
    SIMILARITY_LANE_WEIGHTS,
    SIMILARITY_LOW_COHESION_FALLBACK,
    SIMILARITY_QDRANT,
    SIMILARITY_QDRANT_FILTER_ACTIVE,
    SIMILARITY_QDRANT_HIT_COUNT,
    SIMILARITY_QDRANT_HITS_BY_SPACE,
    SIMILARITY_QDRANT_LIMIT_PER_SPACE,
    SIMILARITY_QDRANT_PROBE_KIND,
    SIMILARITY_QDRANT_REQUESTED_COUNT,
    SIMILARITY_QDRANT_RETURNED_COUNT,
    SIMILARITY_QDRANT_SPACE_COUNT,
    SIMILARITY_QDRANT_SPACES,
    SIMILARITY_RETRIEVAL_LANES,
    SIMILARITY_RETRIEVAL_TOTAL,
    SIMILARITY_SHAPE_MODIFIERS,
    SIMILARITY_VECTOR_SPACE_COHESION,
    SIMILARITY_VECTOR_SPACE_WEIGHTS,
    SIMILARITY_WEAVE_TARGETS,
)

# Manual instrumentation tracer. A no-op proxy when `setup_tracing` hasn't run
# (offline / the tmdb-facing debug entrypoint / tests), so every span/attr below
# is a cheap no-op outside a traced request.
tracer = trace.get_tracer(__name__)


LaneName = Literal[
    "shape",
    "director",
    "franchise",
    "studio",
    "source",
    "quality",
    "format",
    "themes",
    "cast",
    "specific_award",
]

# Lanes whose scores feed the additive sum. Studio is intentionally
# excluded — V2 made it a multiplier on shape-qualifying candidates so
# Blumhouse-style on-brand-but-low-shape noise no longer crowds the top
# of the list. The themes/cast/specific_award lanes are multi-anchor
# only; their weights collapse to 0 in single-anchor flow so the same
# pipeline handles both cases without branching.
#
# V3: director is passthrough (raw = absolute contribution); it
# participates in the additive sum but bypasses the `_normalize_weights`
# denominator. See PASSTHROUGH_LANES below.
#
# V3.4.4: rare_keyword lane removed entirely. Tag-overlap scoring is
# fully absorbed into the themes lane (with a new compounding bonus
# matching the old combo bonus). Truly-rare match exploration runs
# through the rare_keyword bucket in the greedy weaver instead.
#
# V3.4.8: director removed from the additive sum entirely. The
# director lane is now a multiplier on shape-qualifying candidates
# (mirroring the studio multiplier pattern), so same-director matches
# get amplified when shape similarity is real and barely move when
# it isn't. This replaces the V3.4.7 director fatigue gate, which
# was banning genuinely-good auteur matches alongside the bad ones.
ADDITIVE_LANES: tuple[LaneName, ...] = (
    "shape",
    "franchise",
    "source",
    "quality",
    "format",
    "themes",
    "cast",
    "specific_award",
)

# Lanes whose raw score IS the absolute contribution (no weight scaling).
# They appear in the additive sum but are excluded from the normalization
# denominator so adding them doesn't dilute the proportional lanes.
# V3.4.8: empty — director was the only passthrough lane, and it's now
# a multiplier rather than additive. Kept for future passthrough lanes.
PASSTHROUGH_LANES: frozenset[LaneName] = frozenset()

# All lanes carried through the debug payload, including the studio
# multiplier so debug output still shows when on-brand candidates were
# scored. Order matches LaneName.
ALL_LANES: tuple[LaneName, ...] = (
    "shape",
    "director",
    "franchise",
    "studio",
    "source",
    "quality",
    "format",
    "themes",
    "cast",
    "specific_award",
)

AnchorType = Literal[
    "standard_shape",
    "cult_garbage",
    "prestige",
    "franchise_dominant",
    "studio_lineage",
    "source_material",
    "director_signature",
]

# V2 single-anchor base vector-space weights — drops the V1 tier
# grouping. Higher narrative_techniques (storytelling-style binds
# Tarantino, Nolan), lower production (V1 locked Titanic to ship-disaster
# matches), and plot_events stays the lowest signal.
VECTOR_BASE_WEIGHTS_SINGLE: dict[VectorName, float] = {
    VectorName.PLOT_ANALYSIS: 1.00,
    VectorName.VIEWER_EXPERIENCE: 1.00,
    VectorName.WATCH_CONTEXT: 0.75,
    VectorName.NARRATIVE_TECHNIQUES: 0.55,
    VectorName.RECEPTION: 0.55,
    VectorName.ANCHOR: 0.45,
    VectorName.PRODUCTION: 0.30,
    VectorName.PLOT_EVENTS: 0.25,
}

# V2 multi-anchor base vector weights — kept at the V1 tiered set
# because cohesion does the heavy lifting and V1 results on cohesive
# anchor sets (Pixar, Ghibli, war films, Tarantino) were already strong.
VECTOR_BASE_WEIGHTS_MULTI: dict[VectorName, float] = {
    VectorName.PLOT_ANALYSIS: 1.00,
    VectorName.VIEWER_EXPERIENCE: 1.00,
    VectorName.WATCH_CONTEXT: 0.65,
    VectorName.PRODUCTION: 0.65,
    VectorName.RECEPTION: 0.65,
    VectorName.ANCHOR: 0.65,
    VectorName.NARRATIVE_TECHNIQUES: 0.35,
    VectorName.PLOT_EVENTS: 0.35,
}

BASE_LANE_WEIGHTS: dict[LaneName, float] = {
    "shape": 0.60,
    # V3.4.8: director moved from passthrough to multiplier path.
    # Kept here at 0 weight so debug output still surfaces the raw
    # director score, but it contributes 0 to the additive sum. The
    # multiplier path (DIRECTOR_MULTIPLIER_*) handles its actual
    # influence on combined score.
    "director": 0.00,
    "franchise": 0.12,
    "studio": 0.06,        # debug-only; the multiplier doesn't read it
    "source": 0.04,
    "quality": 0.06,
    "format": 0.04,
    # Multi-only lanes; single-anchor zeros these out.
    # V3.1: bumped from 0.06 → 0.12 — the V3 smoke run showed themes
    # raw scores of 0.20–0.95 on genuinely-themed candidates contribute
    # only 0.012–0.057 at weight 0.06, sub-perceptual against shape
    # (~0.50 weight). 0.12 doubles thematic recall without tipping
    # Pixar / MCU / Star-Wars cohesion (still shape-dominated).
    "themes": 0.12,
    "cast": 0.03,
    "specific_award": 0.04,
}

# Single-anchor weight deltas applied additively per active anchor type.
# studio_lineage is retained as a debug flag with no weight delta — V2
# studio handling is multiplicative, not additive.
#
# V3: director_signature also has no delta. The V3 director lane is a
# passthrough that contributes 0.20 in absolute terms when an auteur
# match fires, so the V2 +0.10 weight bump is redundant. The anchor
# type still appears in `active_anchor_types` for debug visibility.
#
# source_material delta dropped from V1 +0.14 to V2 +0.08 because the
# IDF-weighted source lane no longer over-credits "novel"-tier matches.
SINGLE_ANCHOR_ADJUSTMENTS: dict[AnchorType, dict[LaneName, float]] = {
    "standard_shape": {},
    "cult_garbage": {"quality": 0.26, "shape": -0.10},
    "prestige": {"quality": 0.16, "shape": -0.06},
    "franchise_dominant": {"franchise": 0.18, "shape": -0.08},
    "studio_lineage": {},
    "source_material": {"source": 0.08, "shape": -0.04},
    "director_signature": {},
}


COMPETITIVE_BAND = 0.08
TOP_SECTION_SIZE = 10
TOP_FORMAT_LOCK = 5             # top-5 must share the anchor format bucket
MAX_TOP_DOMINANT_LANE = 4
MAX_TOP_FRANCHISE = 3

# V3.4 Bucket-Weaver constants. The weaver replaces the V3 dominance /
# franchise / competitive-band caps inside `_weave_candidates` with an
# explicit slot-allocation + greedy-MMR pass. Per-bucket cap (3) and
# best-overall floor (5) together prevent any single bucket from
# saturating the top section. λ=0.5 balances relevance vs. starvation.
# See similar_movies.md §V3.4 for full design rationale.
WEAVER_TOTAL_SLOTS            = 10
WEAVER_BEST_OVERALL_FLOOR     = 5
WEAVER_BUCKET_CAP             = 3
# V3.4.1: bumped 0.30 → 0.50 to require ≥half-anchor cohesion before
# instantiating signal buckets. Eliminates Slasher trio's auteur firing
# at 1/3 cohesion and rare_keyword firing on heterogeneous one-anchor
# outlier traits. Single-anchor signals are unaffected (auteur=1.0,
# franchise=catalog×max_pop typically clears 0.50).
WEAVER_BUCKET_INSTANTIATE_MIN = 0.50
WEAVER_LAMBDA                 = 0.5

# Franchise bucket gating: candidates with structural score < 0.55 are
# tier-4 universe-only (e.g., Iron Man ↔ Eternals, Phase-1 ↔ Phase-4
# MCU with drifted subgroups). Excluded from the franchise bucket per
# V3.4 Decision 6 to avoid the "all Marvel movies ever" failure mode.
WEAVER_FRANCHISE_BUCKET_MIN_SCORE = 0.55

# Rare-keyword bucket membership: a candidate qualifies when a shared
# anchor trait has IDF >= this threshold (matching the high-tier cutoff
# in the rare-keyword scoring lane). Candidates with only moderate-tier
# matches don't qualify for the bucket — those rely on the unified
# ranker's themes / rare_keyword lane contributions, not a separate row.
WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN = 0.55

# V3.4.6 franchise fatigue (single-anchor only). Strong franchise anchors
# (John Wick, Star Wars, MCU, etc.) tend to flood the result list with
# their own sequels because shape similarity + the franchise lane stack
# additively. The user typically wants *some* franchise siblings but
# also genuine non-franchise alternatives.
#
# Mechanism: in the greedy weaver, after each placement, track the ratio
# of placed franchise entries to placed non-franchise entries. While
# `franchise_count > FRANCHISE_FATIGUE_THRESHOLD * non_franchise_count`
# the weaver hard-bans franchise candidates from every bucket and falls
# through to the next non-franchise pick. At threshold 0.34 this caps
# franchise representation at roughly 1:3 (≤25% of the top section).
#
# "Franchise" here is defined broadly: candidate's
# `(lineage_entry_ids ∪ shared_universe_entry_ids)` intersects the
# anchor's `(lineage_entry_ids ∪ shared_universe_entry_ids)` — any
# cross-match between either kind on either side qualifies, regardless
# of subgroup or crossover/spinoff tags. Multi-anchor flow is unaffected:
# multi-anchor consensus on franchise membership is real signal, not
# stacking artifact.
FRANCHISE_FATIGUE_THRESHOLD = 0.34

BUCKET_BEST_OVERALL = "best_overall"
BUCKET_AUTEUR       = "auteur"
BUCKET_FRANCHISE    = "franchise"
BUCKET_RARE_KEYWORD = "rare_keyword"
BUCKET_LEAD_ACTOR   = "lead_actor"
ALL_BUCKETS: tuple[str, ...] = (
    BUCKET_BEST_OVERALL,
    BUCKET_AUTEUR,
    BUCKET_FRANCHISE,
    BUCKET_RARE_KEYWORD,
    BUCKET_LEAD_ACTOR,
)

# V3 §2.2: V2's franchise-confidence and low-confidence-multiplier paths
# are retired. The structural matrix in `_franchise_score_v2` scores
# every hit additively; "low-confidence" lineages now show up as 0.30
# universe-only or 0.55 same-subgroup-different-lineage scores and
# can't dominate top results without strong shape backing.

# V2 multipliers on combined score (applied post-additive-sum).
STUDIO_MULTIPLIER_SHAPE_GATE = 0.60
STUDIO_MULTIPLIER_STRENGTH = 0.10            # +10% per unit studio_score

# V3.4.8 director multiplier (single-anchor + multi-anchor). Replaces
# the passthrough additive lane. Same shape as studio: candidates
# clearing the shape gate get score *= 1 + STRENGTH * director_score.
# Lower gate (0.30) than studio (0.60) because director is a stronger
# stylistic signal even at moderate shape; larger strength (0.30 vs
# 0.10) because director is the primary "auteur sensibility" signal.
# A full match (director_score=1.0) at gate-clearing shape gets a
# +30% boost; weak-shape candidates with same director don't qualify
# at all, which fixes the Burton-omnibus / Nolan-flood pattern at
# the source rather than after the fact via fatigue.
DIRECTOR_MULTIPLIER_SHAPE_GATE = 0.30
DIRECTOR_MULTIPLIER_STRENGTH = 0.30

# V3.4.9 director gap-boost (rubber-band). The multiplier on its own
# can suppress auteur films too aggressively — TDK loses Nolan films
# past slot 5, but a user looking at "movies like The Dark Knight"
# expects the next Nolan to surface rather than yet another non-Nolan
# crime thriller. The gap-boost tracks how many consecutive
# non-director-match slots have been placed in the top section and
# progressively amplifies the *full score* of any director-match
# candidate (compounding 10% per step past the threshold) so a
# director match can climb back into contention.
#
# Mechanism: when ``enforce_director_gap_boost`` is set (single-anchor
# flow only), the weaver tracks ``consecutive_non_director``. At each
# slot, compute ``k = max(0, gap - DIRECTOR_GAP_THRESHOLD + 1)``. If
# k > 0, any director-match candidate's effective score for that slot's
# MMR comparison is multiplied by ``(1 + DIRECTOR_GAP_INCREMENT) ** k``.
# The boost has no shape gate — it applies to every candidate sharing
# a director with the anchor (independent of whether the V3.4.8
# multiplier fired). The counter resets the moment a director-match
# film is placed.
#
# Worked example: director placed at slot 1, non-director at slots 2,
# 3, 4. Entering slot 5, gap = 3 → k = 1 → boost = 1.10. Still
# non-director → slot 6 sees gap = 4 → k = 2 → boost = 1.21. And so
# on (1.331, 1.464, …) until the next director-match placement, which
# resets the counter back to 0.
DIRECTOR_GAP_THRESHOLD = 3
DIRECTOR_GAP_INCREMENT = 0.10

# Medium multiplier: V3 §3.2 piecewise — cross-category (live-action ↔
# animation, where MEDIUM_SIMILARITY returns 0.0) gets a hard 0.65× hit;
# within-category mismatches (CG ↔ stop-motion, etc.) keep the V2 formula
# 0.85 + 0.15 * score, which still floors at 0.85 for cross-anim-style
# crossings but rewards perfect agreement at 1.00. The cross-category
# 0.65 fixes the Dark Knight Rises case where animated Batman entries
# (Year One, Long Halloween) were riding the V2 0.85 floor into the top
# 10 — H4 in the V3 hypothesis table.
MEDIUM_MULTIPLIER_FLOOR = 0.85
MEDIUM_MULTIPLIER_RANGE = 0.15
MEDIUM_CROSS_CATEGORY_MULTIPLIER = 0.65

# V3.4.2 format-mismatch multiplier. Symmetric with the medium piecewise
# multiplier but harsher: the format buckets (narrative_feature /
# documentary / short / performance / news / tv_format) are categorical
# *content types*, not style variations within the same form. A
# documentary about Barbie is fundamentally not "a movie like Barbie"
# in a way that animated-vs-live-action animation simply isn't. Lane-
# level format scoring already gates the top-5 lock; this multiplier
# extends the penalty across all 10 slots so cross-format candidates
# can't ride high-V3 scores into slots 6–10 (the Barbie meta-doc case).
# Mockumentary was deliberately removed from the format taxonomy and
# is treated as a high-signal theme keyword instead — narrative-styled
# mockumentaries (Spinal Tap, What We Do in the Shadows) shouldn't
# eat this multiplier when the anchor is a narrative feature.
FORMAT_CROSS_CATEGORY_MULTIPLIER = 0.35

# Country/language coherence multiplier. V3 §2.4: extended to single-
# anchor (V2 was multi-only) and recalibrated. The looser V2 boost
# (1.10) was too generous for "you got the country right" given how
# weak country alone is as a similarity signal; the looser V2 penalty
# (0.85) was too soft for the Barbie-Telugu-#1 failure case.
COUNTRY_CONSENSUS_BOOST = 1.05
COUNTRY_CONSENSUS_PENALTY = 0.75

# V3.3: Shape multiplier — identity-level boost for candidates that
# share the anchor's reach × quality "shape." Five named shapes derived
# from imdb_vote_count tier × _quality_bucket(row); see
# similar_movies.md §V3.3 for the full grid + design rationale.
#
# STRONG shapes carry a sharply distinctive identity (cult, dogshit,
# hidden_gem); MODERATE shapes (prestige, mainstream_blockbuster)
# overlap heavily with general cinema and so deserve a smaller lift.
SHAPE_REACH_HIGH_THRESHOLD = 100_000     # ≥100K vote_count → HIGH zone
SHAPE_REACH_LOW_THRESHOLD  = 10_000      # <10K vote_count   → LOW zone
SHAPE_BOOST_STRONG    = 0.15             # max ×1.15 for cohesion 1.0
SHAPE_BOOST_MODERATE  = 0.08             # max ×1.08 for cohesion 1.0
SHAPE_COHESION_MIN    = 0.50             # multi-anchor: cohort needs
                                          # ≥50% of anchors carrying the
                                          # shape for the boost to fire

# V3.3.1/V3.3.2: Shape classification thresholds — independent of
# `_quality_bucket` (which drives the legacy prestige/cult quality
# scoring formulas and is conservatively gated). Decoupling lets the
# shape multiplier participate on indie / cult / dogshit films without
# altering legacy lane behavior.
#
# V3.3.2 raised the default prestige floor from 78 → 80 (single
# award-less reception data is too noisy a prestige signal at 78);
# added an award-aware lowered floor at 65 that fires only when a
# film carries a *picture-level* signal (Best Picture or Director
# nom/win at non-Razzie ceremony — see SHAPE_PICTURE_LEVEL_TAG_IDS
# in db/postgres.py). Acting and craft category awards explicitly
# don't elevate (Bohemian Rhapsody won 4 Oscars in performance/craft
# categories; the *film* isn't prestige). Multi-ceremony win count
# was considered but dropped — too many films pick up a win in
# craft categories at multiple ceremonies without being prestige.
#
# The Razzie side: any Razzie WIN in a specific WORST_* category
# (excluding WORST_OTHER which might catch the Razzie Redeemer Award)
# raises the poorly-rated ceiling from 50 to 60. Razzie nominations
# alone don't shift; only WINS count for the elevation.
SHAPE_PRESTIGE_RECEPTION_MIN          = 80.0   # default (V3.3.2: 78.0 → 80.0)
SHAPE_PRESTIGE_RECEPTION_MIN_W_AWARD  = 65.0   # with picture-level signal
SHAPE_POOR_RECEPTION_MAX              = 50.0
SHAPE_POOR_RECEPTION_MAX_W_RAZZIE     = 60.0   # with bad-Razzie WIN
# Percentile gates dropped — the reach axis (vote-count tiers) already
# filters by audience size, so a separate percentile floor would just
# duplicate that signal and exclude legitimate dogshit/cult candidates
# (Mega Python at percentile 0.787, Leprechaun 4 at 0.883 both fail
# the legacy 0.89 cult cutoff).

SHAPE_DOGSHIT                = "dogshit"
SHAPE_CULT_GARBAGE           = "cult_garbage"
SHAPE_PRESTIGE               = "prestige"
SHAPE_HIDDEN_GEM             = "hidden_gem"
SHAPE_MAINSTREAM_BLOCKBUSTER = "mainstream_blockbuster"

SHAPE_STRENGTHS: dict[str, float] = {
    SHAPE_DOGSHIT:                SHAPE_BOOST_STRONG,
    SHAPE_CULT_GARBAGE:           SHAPE_BOOST_STRONG,
    SHAPE_HIDDEN_GEM:             SHAPE_BOOST_STRONG,
    SHAPE_PRESTIGE:               SHAPE_BOOST_MODERATE,
    SHAPE_MAINSTREAM_BLOCKBUSTER: SHAPE_BOOST_MODERATE,
}

# V3.3.2: Cross-bucket shape multiplier matrix. Strength on [0, 1];
# multiplied by the candidate's shape's max strength (STRONG 0.15 /
# MODERATE 0.08) and the cohort cohesion to compute the effective
# boost. Missing pairs default to 0.0 (no cross-shape boost).
#
# Same-shape pairs are 1.0 (identity boost). The 0.7 pairs are
# "boundary-arbitrary same-quality reach splits" (prestige/hidden_gem,
# cult_garbage/dogshit) — the system's reach cutoff at 10K is
# arbitrary and the audience overlap is huge. The 0.4/0.25/0.15/0.10
# pairs are quality-step crossings via the mainstream blockbuster
# bridge — they don't fire on single-anchor under SHAPE_COHESION_MIN
# (0.5) but DO fire on mixed cohorts where same-shape and cross-shape
# contributions sum to ≥0.5.
SHAPE_CROSS_STRENGTH: dict[tuple[str, str], float] = {
    # Same-shape (identity)
    (SHAPE_PRESTIGE,               SHAPE_PRESTIGE):               1.0,
    (SHAPE_HIDDEN_GEM,             SHAPE_HIDDEN_GEM):             1.0,
    (SHAPE_MAINSTREAM_BLOCKBUSTER, SHAPE_MAINSTREAM_BLOCKBUSTER): 1.0,
    (SHAPE_CULT_GARBAGE,           SHAPE_CULT_GARBAGE):           1.0,
    (SHAPE_DOGSHIT,                SHAPE_DOGSHIT):                1.0,
    # 0.7 — same quality, reach split (boundary-arbitrary)
    (SHAPE_PRESTIGE,     SHAPE_HIDDEN_GEM): 0.7,
    (SHAPE_HIDDEN_GEM,   SHAPE_PRESTIGE):   0.7,
    (SHAPE_CULT_GARBAGE, SHAPE_DOGSHIT):    0.7,
    (SHAPE_DOGSHIT,      SHAPE_CULT_GARBAGE): 0.7,
    # 0.4 — same HIGH reach, quality step via mainstream bridge
    (SHAPE_PRESTIGE,               SHAPE_MAINSTREAM_BLOCKBUSTER): 0.4,
    (SHAPE_MAINSTREAM_BLOCKBUSTER, SHAPE_PRESTIGE):               0.4,
    # 0.25 — overlapping reach, quality step (cult ↔ mainstream)
    (SHAPE_CULT_GARBAGE,           SHAPE_MAINSTREAM_BLOCKBUSTER): 0.25,
    (SHAPE_MAINSTREAM_BLOCKBUSTER, SHAPE_CULT_GARBAGE):           0.25,
    # 0.15 / 0.10 — diagonal (reach + quality both differ)
    (SHAPE_MAINSTREAM_BLOCKBUSTER, SHAPE_HIDDEN_GEM): 0.15,
    (SHAPE_HIDDEN_GEM,             SHAPE_MAINSTREAM_BLOCKBUSTER): 0.15,
    (SHAPE_MAINSTREAM_BLOCKBUSTER, SHAPE_DOGSHIT):    0.10,
    (SHAPE_DOGSHIT,                SHAPE_MAINSTREAM_BLOCKBUSTER): 0.10,
    # All other pairs (prestige↔cult/dogshit, hidden_gem↔cult/dogshit)
    # default to 0.0 — opposite quality poles, no audience overlap.
}

# Selective rare-medium retrieval gate (V2 single-anchor): anchor medium
# tags with idf >= this threshold trigger a candidate-pool expansion via
# fetch_movie_ids_by_overall_keywords. LIVE_ACTION is always excluded
# because every catalog entry would qualify.
RARE_MEDIUM_IDF_THRESHOLD = 0.50

# V3 §2.1: director_signature anchor type fires when at least one of
# the anchor's directors is in the curated auteur list. The V2
# `mv_director_strength` percentile gate is retired (it conflated fame
# with stylistic coherence — Lucas surfacing American Graffiti for Star
# Wars, etc.). The auteur list is the canonical source of truth.

# Low-cohesion fallback (multi-anchor): when both vector cohesion and
# every metadata lane's cohesion are weak, the centroid is in noise and
# we fall back to round-robin per-anchor single-anchor results.
# V3 §4.2: raised the metadata threshold from 1.00 to 1.50 because the
# old bar still let "prestige-by-2-of-3" cases (metadata_max_cohesion
# ~1.69) bypass the chaotic-mixed-bag handling.
LOW_COHESION_VECTOR_THRESHOLD = 0.35
LOW_COHESION_METADATA_MAX_THRESHOLD = 1.50

# V3.1: bumped from 500 → 2000 to widen the shape-recall funnel.
# Lady-Bird-vs-Barbie style auteur/themes matches whose vector
# embedding doesn't align with the anchor's surface aesthetic were
# missing the candidate pool entirely; 4× wider shape pool catches
# them and lets the new themes-recall + director-floor paths score
# them on their actual signal.
DEFAULT_QDRANT_LIMIT = 2000
DEFAULT_QUALITY_LIMIT = 500

NON_RAZZIE_AWARD_IDS: frozenset[int] = frozenset(
    c.ceremony_id for c in AwardCeremony if c is not AwardCeremony.RAZZIE
)

LIVE_ACTION_TAG_ID = OverallKeyword.LIVE_ACTION.keyword_id


@dataclass(frozen=True, slots=True)
class LaneEvidence:
    lane_scores: dict[LaneName, float]
    candidate_sources: list[LaneName]
    dominant_lane: LaneName
    # V3 diagnostic fields — populated by `_build_results` so debug
    # consumers (batch runner, analysis scripts) can see how the final
    # score was assembled. `base_score` is the additive sum BEFORE
    # multipliers / floors. `multipliers` only contains keys for
    # multipliers actually applied (a value of 1.0 means "applied as
    # identity" — e.g., medium with same-medium match). `floor_value`
    # > 0 indicates a floor displaced the additive computation;
    # `floor_source` names the lane it came from. Defaults keep
    # backwards compatibility with any caller that constructs a
    # LaneEvidence without diagnostics.
    base_score: float = 0.0
    multipliers: dict[str, float] = field(default_factory=dict)
    floor_value: float = 0.0
    floor_source: str = ""


@dataclass(frozen=True, slots=True)
class SimilarMovieResult:
    movie_id: int
    score: float
    evidence: LaneEvidence


@dataclass(frozen=True, slots=True)
class SimilarMoviesDebug:
    vector_space_weights: dict[str, float]
    vector_space_cohesion: dict[str, float] = field(default_factory=dict)
    raw_lane_weights: dict[LaneName, float] = field(default_factory=dict)
    normalized_lane_weights: dict[LaneName, float] = field(default_factory=dict)
    candidate_counts_by_lane: dict[LaneName, int] = field(default_factory=dict)
    # Retrieval-side counts (distinct from the scoring-side
    # `candidate_counts_by_lane` above): {lane: result_count} for every
    # candidate-fetch query that actually ran (seed non-empty), and the
    # deduped union size. Powers the branch span's `branch_retrieval_lanes` /
    # `branch_retrieval_total`. A fired-but-empty lane is present at 0; a
    # gated-off lane is absent.
    retrieval_counts_by_lane: dict[str, int] = field(default_factory=dict)
    retrieval_total: int = 0
    # Multi-anchor per-metadata-lane cohort cohesion (`cohesion_by_lane`),
    # surfaced for the branch span's `branch_lane_cohesion`. Empty single-anchor.
    lane_cohesion: dict[str, float] = field(default_factory=dict)
    # V2 additions — non-additive signals and audit trails. All optional so
    # a single-anchor flow can leave the multi-anchor-only fields empty
    # without forcing callers to deal with `None` checks.
    anchor_format_bucket: FormatBucket | None = None
    anchor_medium_tags: list[int] = field(default_factory=list)
    # Anchor reach×quality shape cohesion: {shape: M_s/N}. Single-anchor is
    # {shape: 1.0} (or empty when the anchor is shapeless); multi-anchor
    # carries the per-shape cohort fraction. Surfaced so the branch span can
    # record the "obscure vs blockbuster" axis that active_anchor_types can't.
    anchor_shape_cohesion: dict[str, float] = field(default_factory=dict)
    franchise_high_confidence: bool = False
    consensus_countries: list[int | str] = field(default_factory=list)
    low_cohesion_fallback_used: bool = False
    per_anchor_active_anchor_types: dict[int, list[AnchorType]] = field(
        default_factory=dict
    )
    # V3 multi-anchor diagnostic: when ≥50% of anchors are shorts,
    # `_run_multi_anchor_similarity` flips into shorts-friendly mode
    # (boost on, downrank off). Surfaced here so the batch runner can
    # explain why short candidates are scoring high.
    shorts_dominant: bool = False


@dataclass(frozen=True, slots=True)
class SimilarMoviesSearchResult:
    anchor_movie_ids: list[int]
    ranked: list[SimilarMovieResult]
    active_anchor_types: list[AnchorType]
    debug: SimilarMoviesDebug


@dataclass(frozen=True, slots=True)
class _ResolvedStudioEntry:
    company_id: int
    entry: StudioSimilarityEntry


@dataclass(slots=True)
class _CandidateScore:
    movie_id: int
    score: float
    lane_scores: dict[LaneName, float]
    candidate_sources: list[LaneName]
    dominant_lane: LaneName
    # Diagnostic mirrors of the LaneEvidence diagnostic fields. Carried
    # through weaving so the eventual SimilarMovieResult.LaneEvidence
    # can pass them out unchanged.
    base_score: float = 0.0
    multipliers: dict[str, float] = field(default_factory=dict)
    floor_value: float = 0.0
    floor_source: str = ""
    # V3.4.9 gap-boost input. ``director_match`` is True for any
    # candidate sharing a director with the anchor (independent of
    # whether shape clears the V3.4.8 multiplier gate). The weaver
    # uses it both to (a) reset its consecutive-non-director counter
    # on placement and (b) decide which candidates' effective scores
    # get the compounding rubber-band multiplier in MMR comparison.
    director_match: bool = False


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _release_year(row: dict) -> int | None:
    release_ts = row.get("release_ts")
    if release_ts is None:
        return None
    return datetime.fromtimestamp(release_ts, tz=timezone.utc).year


def _as_int_set(value: object) -> set[int]:
    if not value:
        return set()
    return {int(v) for v in value}


def _normalize_weights(raw: dict[LaneName, float]) -> dict[LaneName, float]:
    """Normalize raw lane weights so the proportional additive lanes sum to 1.0.

    V3: the director passthrough lane carries an absolute contribution
    defined by its scoring function, so it's excluded from both the
    denominator AND the per-lane normalization. Its raw weight flows
    through unchanged (typically 1.0 = identity multiplier). Studio is
    excluded for the V2 reason (multiplier-only).

    Negative weights (possible after applying anchor-type deltas) get
    clamped to 0 before the division. If every proportional lane is
    zero or negative, fallback is "all weight on shape" so the flow
    still returns ranked results.
    """
    proportional_lanes = tuple(
        lane for lane in ADDITIVE_LANES if lane not in PASSTHROUGH_LANES
    )
    total = sum(max(raw.get(lane, 0.0), 0.0) for lane in proportional_lanes)
    if total <= 0.0:
        # Fallback when every proportional lane is zero or negative —
        # collapse to "all weight on shape" so the flow still returns
        # ranked results. Other proportional lanes get 0; passthrough
        # lanes still flow through with their raw weight.
        out: dict[LaneName, float] = {lane: 0.0 for lane in proportional_lanes}
        out["shape"] = 1.0
    else:
        out = {
            lane: max(raw.get(lane, 0.0), 0.0) / total
            for lane in proportional_lanes
        }
    # Passthrough lanes carry their raw weight unchanged (typically 1.0).
    for lane in PASSTHROUGH_LANES:
        out[lane] = max(raw.get(lane, 0.0), 0.0)
    # Studio carries through unnormalized so debug output preserves the raw
    # studio score; the multiplier doesn't read this field.
    out["studio"] = max(raw.get("studio", 0.0), 0.0)
    return out


def _normalize_vector_weights(
    raw: dict[VectorName, float],
) -> dict[VectorName, float]:
    total = sum(raw.values())
    if total <= 0.0:
        return {}
    return {space: weight / total for space, weight in raw.items()}


def _normalize_lane_to_max(scores: dict[int, float]) -> dict[int, float]:
    """Rescale a merged lane so its strongest candidate is 1.0."""
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0.0:
        return {}
    return {movie_id: _clamp(score / max_score) for movie_id, score in scores.items()}


def _quality_bucket(row: dict) -> str:
    reception = row.get("reception_score")
    percentile = row.get("popularity_percentile") or 0.0
    if reception is not None and reception <= 45 and percentile >= 0.89:
        return "cult_garbage"
    if reception is not None and reception >= 85 and percentile >= 0.75:
        return "prestige"
    return "middle"


def _classify_shape(
    row: dict,
    award_signal: "SimilarityAwardSignals | None" = None,
) -> str | None:
    """V3.3.2: classify a movie into one of 5 reach × quality shapes.

    Returns one of SHAPE_DOGSHIT / SHAPE_CULT_GARBAGE / SHAPE_PRESTIGE /
    SHAPE_HIDDEN_GEM / SHAPE_MAINSTREAM_BLOCKBUSTER, or None for films in
    shapeless cells ("just normal" films with no identity-level signal).

    V3.3.2 award-aware thresholds:
      - Prestige floor is `SHAPE_PRESTIGE_RECEPTION_MIN` (80) by default,
        lowered to `SHAPE_PRESTIGE_RECEPTION_MIN_W_AWARD` (65) when the
        film carries a *picture-level* signal (Best Picture or Director
        nom/win at any non-Razzie ceremony). Acting/craft awards alone
        don't elevate; the signal is film-level only.
      - Poor ceiling is `SHAPE_POOR_RECEPTION_MAX` (50) by default,
        raised to `SHAPE_POOR_RECEPTION_MAX_W_RAZZIE` (60) when the film
        won a Razzie in a specific WORST_* category (excluding
        WORST_OTHER which may catch the positive Razzie Redeemer Award).
      - Award signals can also classify a film as prestige or poor when
        reception data is missing entirely.

    Conflict resolution: if a film has both a picture-level signal AND
    a bad Razzie win (rare), Razzie wins at low reception (≤60) and
    picture-level wins at mid-to-high reception (65–79). The branches
    below are ordered so Razzie elevation runs first and only fires on
    appropriately-low reception.
    """
    reception = row.get("reception_score")
    votes = row.get("imdb_vote_count") or 0
    has_picture_level = (
        award_signal is not None and award_signal.has_picture_level_signal
    )
    has_bad_razzie_win = (
        award_signal is not None and award_signal.has_bad_razzie_win
    )

    poor_ceiling = (
        SHAPE_POOR_RECEPTION_MAX_W_RAZZIE
        if has_bad_razzie_win
        else SHAPE_POOR_RECEPTION_MAX
    )
    prestige_floor = (
        SHAPE_PRESTIGE_RECEPTION_MIN_W_AWARD
        if has_picture_level
        else SHAPE_PRESTIGE_RECEPTION_MIN
    )

    if reception is None:
        # No reception data — fall back to award signals if present.
        # Razzie win wins ties because a Razzie is more distinctive
        # than "the film won an Oscar but we have no critical reception".
        if has_bad_razzie_win:
            return (
                SHAPE_DOGSHIT
                if votes < SHAPE_REACH_LOW_THRESHOLD
                else SHAPE_CULT_GARBAGE
            )
        if has_picture_level:
            return (
                SHAPE_HIDDEN_GEM
                if votes < SHAPE_REACH_LOW_THRESHOLD
                else SHAPE_PRESTIGE
            )
        # No signal of any kind — only HIGH reach earns mainstream
        # blockbuster (it's known by sheer audience size).
        return (
            SHAPE_MAINSTREAM_BLOCKBUSTER
            if votes >= SHAPE_REACH_HIGH_THRESHOLD
            else None
        )

    if reception <= poor_ceiling:
        # Poorly rated films at LOW reach are "dogshit" (just bad);
        # at HIGH or MID reach they're "cult_garbage" (loved-for-badness).
        return (
            SHAPE_DOGSHIT
            if votes < SHAPE_REACH_LOW_THRESHOLD
            else SHAPE_CULT_GARBAGE
        )
    if reception >= prestige_floor:
        # Acclaimed films at LOW reach are "hidden_gem"; at HIGH/MID
        # they're standard "prestige" cinema.
        return (
            SHAPE_HIDDEN_GEM
            if votes < SHAPE_REACH_LOW_THRESHOLD
            else SHAPE_PRESTIGE
        )
    # Reception falls in the middle band — neither poor nor acclaimed
    # under the active thresholds. Only HIGH reach earns
    # mainstream_blockbuster; MID and LOW middle-quality films are
    # shapeless.
    if votes >= SHAPE_REACH_HIGH_THRESHOLD:
        return SHAPE_MAINSTREAM_BLOCKBUSTER
    return None


def _shape_multiplier(
    anchor_shape_cohesion: dict[str, float],
    candidate_shape: str | None,
) -> float:
    """V3.3.2: shape boost with cross-bucket support.

    `anchor_shape_cohesion` maps shape → cohesion in [0, 1]:
      - Single-anchor: {anchor.shape: 1.0} if anchor has a shape, else {}.
      - Multi-anchor: {shape: M_s/N for each shape with at least one
        anchor carrying it} where M_s is the count of anchors in that
        shape and N is the cohort size.

    V3.3.2 introduces cross-bucket boosts via the `SHAPE_CROSS_STRENGTH`
    matrix. Effective cohesion sums each anchor shape's contribution
    multiplied by its cross-strength to the candidate's shape. The
    SHAPE_COHESION_MIN gate (0.5) still applies to the *summed* effective
    cohesion — so single-anchor cross-pairs at 0.4 don't fire alone, but
    mixed cohorts where same-shape (1.0) + cross-shape (0.4) contributions
    sum to ≥0.5 do fire (this is what catches the V3.3.1 MCU regression
    on Endgame's prestige boost).

    Returns 1.0 (no boost) when:
      - candidate has no shape (shapeless cell), or
      - effective cohesion is below SHAPE_COHESION_MIN.

    Otherwise returns 1.0 + max_strength[candidate.shape] * effective_cohesion.
    """
    if candidate_shape is None:
        return 1.0
    effective_cohesion = sum(
        cohesion
        * SHAPE_CROSS_STRENGTH.get((anchor_shape, candidate_shape), 0.0)
        for anchor_shape, cohesion in anchor_shape_cohesion.items()
    )
    if effective_cohesion < SHAPE_COHESION_MIN:
        return 1.0
    return 1.0 + SHAPE_STRENGTHS[candidate_shape] * effective_cohesion


def _major_award_win_score(row: dict) -> float:
    wins = _as_int_set(row.get("award_ceremony_win_ids"))
    return 1.0 if wins & NON_RAZZIE_AWARD_IDS else 0.0


def _single_anchor_lane_weights(
    active_anchor_types: list[AnchorType],
    *,
    quality_bucket: str | None = None,
) -> tuple[dict[LaneName, float], dict[LaneName, float]]:
    """Build raw + normalized lane weights for single-anchor flow.

    V3 changes vs. V2: the V2 ``franchise_low_confidence`` gating that
    suppressed the ``franchise_dominant`` adjustment and zeroed the
    franchise lane is gone. The structural V3 franchise matrix already
    scales scores by overlap quality (1.00 same lineage + same subgroup,
    0.30 universe-only, etc.), so direct-to-DVD spinoffs can't ride a
    full +0.18 weight into the top of the list. Themes / cast /
    specific_award stay zeroed in single-anchor flow because those
    lanes are multi-anchor-only signals.
    """
    raw = dict(BASE_LANE_WEIGHTS)
    # V3 §2.3 enables themes in single-anchor flow; cast and
    # specific_award stay multi-only.
    raw["cast"] = 0.0
    raw["specific_award"] = 0.0

    # V3 §4.1: middle-bucket quality weight is halved (0.06 → 0.03).
    # Prestige/cult buckets keep the V2 weight because the per-bucket
    # formulas are bucket-specific and tightly tuned.
    if quality_bucket == "middle":
        raw["quality"] = 0.03

    for anchor_type in active_anchor_types:
        for lane, delta in SINGLE_ANCHOR_ADJUSTMENTS[anchor_type].items():
            raw[lane] = raw.get(lane, 0.0) + delta

    normalized = _normalize_weights(raw)
    return raw, normalized


def _metadata_cohesion(trait_sets: list[set[object]]) -> float:
    if not trait_sets:
        return 0.0
    counts: dict[object, int] = {}
    for traits in trait_sets:
        for trait in traits:
            counts[trait] = counts.get(trait, 0) + 1
    if not counts:
        return 0.0
    max_count = max(counts.values())
    if max_count < 2:
        return 0.0
    repetition_ratio = max_count / len(trait_sets)
    return 2.0 * math.log1p(9.0 * repetition_ratio) / math.log1p(9.0)


# Module-level cache + lock for the studio registry → company_id lookup.
# The mapping is built from `studio_entries_by_normalized_string()` (static,
# in-process) joined to `production_company` via a Postgres lookup that
# never changes within a process lifetime, so we resolve it exactly once.
# An async lock guards the first-call materialization so concurrent
# requests on a cold process don't all fire the same Postgres query.
_studio_entries_cache: dict[int, list[StudioSimilarityEntry]] | None = None
_studio_entries_lock: asyncio.Lock | None = None


async def _load_studio_entries_by_company_id() -> dict[int, list[StudioSimilarityEntry]]:
    global _studio_entries_cache, _studio_entries_lock
    if _studio_entries_cache is not None:
        return _studio_entries_cache
    # Lazily create the lock so module import doesn't bind to a particular
    # event loop. asyncio.Lock() requires a running loop in 3.10+.
    if _studio_entries_lock is None:
        _studio_entries_lock = asyncio.Lock()
    async with _studio_entries_lock:
        # Double-check after acquiring the lock — another task may have
        # already populated the cache while we were waiting.
        if _studio_entries_cache is not None:
            return _studio_entries_cache
        by_norm = studio_entries_by_normalized_string()
        id_by_norm = await fetch_production_company_ids_by_normalized_strings(
            sorted(by_norm)
        )
        out: dict[int, list[StudioSimilarityEntry]] = {}
        for norm, company_id in id_by_norm.items():
            out.setdefault(company_id, []).extend(by_norm.get(norm, ()))
        _studio_entries_cache = out
        return _studio_entries_cache


def _active_studio_entries(
    row: dict,
    entries_by_company_id: dict[int, list[StudioSimilarityEntry]],
) -> list[_ResolvedStudioEntry]:
    release_year = _release_year(row)
    out: list[_ResolvedStudioEntry] = []
    for company_id in _as_int_set(row.get("production_company_ids")):
        for entry in entries_by_company_id.get(company_id, ()):
            if entry.era.matches(release_year):
                out.append(_ResolvedStudioEntry(company_id=company_id, entry=entry))
    return out


def _studio_score(
    anchor_row: dict,
    candidate_row: dict,
    anchor_entries: list[_ResolvedStudioEntry],
    candidate_entries: list[_ResolvedStudioEntry],
) -> float:
    if not anchor_entries or not candidate_entries:
        return 0.0

    anchor_by_company = {entry.company_id: entry for entry in anchor_entries}
    anchor_year = _release_year(anchor_row)
    candidate_year = _release_year(candidate_row)
    best = 0.0

    for candidate_entry in candidate_entries:
        anchor_entry = anchor_by_company.get(candidate_entry.company_id)
        if anchor_entry is None:
            continue
        score = min(anchor_entry.entry.base_score, candidate_entry.entry.base_score)
        if (
            (anchor_entry.entry.era_sensitive or candidate_entry.entry.era_sensitive)
            and anchor_year is not None
            and candidate_year is not None
        ):
            era_gap = abs(candidate_year - anchor_year)
            score *= 0.60 + 0.40 * math.exp(-era_gap / 18.0)
        best = max(best, score)

    return _clamp(best)


def _franchise_traits(row: dict) -> tuple[set[int], set[int], set[int]]:
    return (
        _as_int_set(row.get("lineage_entry_ids")),
        _as_int_set(row.get("shared_universe_entry_ids")),
        _as_int_set(row.get("subgroup_entry_ids")),
    )


def _franchise_score_v2(anchor_row: dict, candidate_row: dict) -> float:
    """V3 franchise lane — simplified 5-tier matrix over (lineage, subgroup, universe).

    The V3 plan §2.2 originally specified a 2D matrix over (role, overlap)
    where role was mainline/spinoff/crossover. Pre-flight DB verification
    confirmed the catalog data doesn't carry a role discriminator —
    crossover films get their own dedicated lineage rather than
    multiple per-hero lineages (e.g., Avengers films use a single
    "Avengers" lineage 1998, not the union of Iron Man / Cap / Thor
    lineages). Star Wars uses the same pattern: every theatrical film
    sits in lineage [3], with subgroups distinguishing trilogies
    (original [1,2], prequel [2,484], sequel [2,8909], Rogue One
    [12533]).

    The structural intent of the V3 matrix carries over to a 5-tier
    rule on what's actually encoded:

    1. **Same lineage AND ≥1 shared subgroup → 1.00** — direct trilogy
       siblings (Star Wars 1977 ↔ Empire share lin [3] AND sub [1,2]).
       Cross-trilogy cases also clear this if they share the saga-wide
       subgroup [2] (1977 ↔ Phantom Menace via sub [2]).

    2. **Same lineage, no shared subgroup → 0.70** — same series but
       a structurally different cluster (1977 ↔ Rogue One: same lin
       [3], no subgroup overlap).

    3. **Different lineage, ≥1 shared subgroup AND shared universe → 0.55** —
       MCU sibling case (Iron Man ↔ Captain America: different
       hero-lineages 440 vs 447, share subgroup [435,437] AND universe
       [436]).

    4. **Different lineage, shared universe only (no subgroup overlap) →
       0.30** — loosely-related universe entries (e.g., late-phase MCU
       entry vs. early-phase MCU entry whose subgroup tags have drifted).

    5. **Disjoint → 0.00**.

    The V2 ``franchise_consistency >= 0.6`` measurement-artifact gate
    that silenced Star Wars in V2 is gone — the matrix scores by
    structure, so Star Wars's "low statistical consistency" is moot.
    The V2 "low-confidence multiplier" path (LOW_CONF_FRANCHISE_*) is
    also retired; the structural matrix gives the score directly.
    """
    a_lin, a_uni, a_sub = _franchise_traits(anchor_row)
    c_lin, c_uni, c_sub = _franchise_traits(candidate_row)

    same_lineage = bool(a_lin and c_lin and (a_lin & c_lin))
    shared_subgroup = bool(a_sub and c_sub and (a_sub & c_sub))
    shared_universe = bool(a_uni and c_uni and (a_uni & c_uni))

    if same_lineage and shared_subgroup:
        return 1.00
    if same_lineage:
        return 0.70
    if shared_subgroup and shared_universe:
        return 0.55
    if shared_universe:
        return 0.30
    return 0.0


# V3 §2.1 director lane — absolute contributions per the unified rule.
# The lane contributes 0.20 in single-anchor when an auteur match
# fires, 0.20 + 0.10*(M/N) in multi-anchor when an auteur match fires
# (M = anchors sharing the auteur director, N = total anchors), and
# 0.10 * (M/N) in multi-anchor cohesion-only when M ≥ 2. Take the max
# over shared directors so a candidate matching multiple shared
# directors gets the strongest signal (the "WA + PTA 2-2 split"
# example from §2.1: each candidate scores via its own M_d).
# V3.4.8: director score functions return [0, 1] so the value can be
# fed directly into the multiplier path (`score *= 1 + STRENGTH * raw`).
# Prior values (0.20 / 0.30 caps) reflected the additive-contribution
# scale; the multiplier path needs a normalized signal. The relative
# structure is preserved: full match = 1.0, partial cohesion scales
# down, non-auteur single-anchor cohesion = 0.
DIRECTOR_SINGLE_ANCHOR_CONTRIBUTION = 1.00
DIRECTOR_MULTI_ANCHOR_AUTEUR_BASE = 0.67
DIRECTOR_MULTI_ANCHOR_AUTEUR_RATIO = 0.33
DIRECTOR_MULTI_ANCHOR_COHESION_RATIO = 0.33
DIRECTOR_FLOOR_RATIO_THRESHOLD = 0.75
DIRECTOR_FLOOR_SHAPE_GATE = 0.30
DIRECTOR_FLOOR_MAGNITUDE = 0.35

# V3.1: single-anchor director floor — mirrors the multi-anchor floor
# but with a softer shape gate (0.20 vs. 0.30). A curated auteur match
# in single-anchor is a strong signal on its own (the user explicitly
# anchored on that filmmaker's work), so the floor doesn't need a
# strong shape backstop. Single-anchor "ratio" is binary: 1.0 when the
# candidate shares any auteur director with the anchor, else 0.0 — so
# the threshold is also 1.0 (any auteur match qualifies). Magnitude
# matches multi.
DIRECTOR_FLOOR_SINGLE_RATIO_THRESHOLD = 1.0
DIRECTOR_FLOOR_SINGLE_SHAPE_GATE = 0.20
DIRECTOR_FLOOR_SINGLE_MAGNITUDE = 0.35


def _single_anchor_director_score(
    candidate_terms: dict[int, set[int]],
    anchor_directors: set[int],
    auteur_term_ids: frozenset[int],
) -> dict[int, float]:
    """V3.4.8 single-anchor director: returns 1.0 when the anchor's
    director is curated AND the candidate shares that director. The
    [0, 1] range feeds the multiplier path in `_build_results` —
    score *= 1 + DIRECTOR_MULTIPLIER_STRENGTH * raw.

    Returns an empty dict when the anchor has no curated director — the
    lane is silent for non-auteur anchors per V3 §2.1 (the user's
    motivation: "when director matters for non-auteurs it's via
    franchise or shape, both of which are already lanes").
    """
    if not anchor_directors:
        return {}
    auteur_intersect = anchor_directors & auteur_term_ids
    if not auteur_intersect:
        return {}
    scores: dict[int, float] = {}
    for movie_id, candidate_dirs in candidate_terms.items():
        if not candidate_dirs:
            continue
        if candidate_dirs & auteur_intersect:
            scores[movie_id] = DIRECTOR_SINGLE_ANCHOR_CONTRIBUTION
    return scores


def _multi_anchor_director_score(
    *,
    candidate_terms: dict[int, set[int]],
    anchor_director_sets: list[set[int]],
    auteur_term_ids: frozenset[int],
    n: int,
) -> tuple[dict[int, float], dict[int, float]]:
    """V3.4.8 multi-anchor director: per-shared-director contribution
    in [0, 1]. The value feeds the multiplier path in `_build_results`.

    For each director ``d`` that the candidate shares with at least
    one anchor, with ``M_d`` = number of anchors that share ``d`` and
    ``N`` = total anchors:

      - if d is curated: contribution = 0.67 + 0.33 * (M_d / N)  (caps 1.00)
      - elif M_d >= 2  : contribution = 0.33 * (M_d / N)         (caps 0.33)
      - else           : 0 (single-anchor cohesion of a non-auteur is silent)

    Returns the max contribution across shared directors so a candidate
    matching multiple shared directors lights up via whichever signal
    is strongest. Also returns ``max_ratio`` (the max ``M_d / N`` for
    each candidate) so ``_build_results`` can apply the §2.1 cohesion
    floor.

    Multi-director independent boost (e.g., 4 anchors split 2-2 between
    Wes Anderson and PTA) is automatic from the max-over-d structure:
    a WA candidate scores via M_WA=2, N=4 → 0.25; a PTA candidate
    scores via M_PTA=2, N=4 → 0.25. Both boosted, neither suppresses
    the other.
    """
    scores: dict[int, float] = {}
    max_ratios: dict[int, float] = {}
    if n <= 0:
        return scores, max_ratios
    for movie_id, candidate_dirs in candidate_terms.items():
        if not candidate_dirs:
            continue
        # M_d for each director shared with at least one anchor.
        m_per_director: dict[int, int] = {}
        for anchor_dirs in anchor_director_sets:
            shared = candidate_dirs & anchor_dirs
            for d in shared:
                m_per_director[d] = m_per_director.get(d, 0) + 1
        if not m_per_director:
            continue
        best_contribution = 0.0
        best_ratio = 0.0
        for d, m_d in m_per_director.items():
            ratio = m_d / n
            if d in auteur_term_ids:
                contrib = (
                    DIRECTOR_MULTI_ANCHOR_AUTEUR_BASE
                    + DIRECTOR_MULTI_ANCHOR_AUTEUR_RATIO * ratio
                )
            elif m_d >= 2:
                contrib = DIRECTOR_MULTI_ANCHOR_COHESION_RATIO * ratio
            else:
                contrib = 0.0
            if contrib > best_contribution:
                best_contribution = contrib
            if ratio > best_ratio:
                best_ratio = ratio
        if best_contribution > 0.0:
            scores[movie_id] = best_contribution
        if best_ratio > 0.0:
            max_ratios[movie_id] = best_ratio
    return scores, max_ratios


def _source_score_idf(
    anchor_source_ids: set[int],
    candidate_source_ids: set[int],
    source_idfs: dict[int, float],
) -> float:
    """V2 source lane: max IDF over the shared source-material types.

    Common types like ``novel`` (high df → low idf) collapse to ~0.20;
    rare types (``video_game``, ``fairy_tale``, ``stage_play``,
    ``comic_book``) stay near 1.0. ``max`` over shared traits — not sum
    — so a single rare match dominates and two common-tag overlaps
    don't compound into a full score.
    """
    if not anchor_source_ids or not candidate_source_ids:
        return 0.0
    shared = anchor_source_ids & candidate_source_ids
    if not shared:
        return 0.0
    return max(source_idfs.get(t, 0.0) for t in shared)


def _quality_score_v2(
    bucket: str,
    candidate_row: dict,
    award_signals: SimilarityAwardSignals | None,
) -> float:
    """V2 quality lane: per-bucket formulas, always-on across buckets.

    cult_garbage: weight low_reception + popularity_match heavily, with
        a 10% boost for any Razzie evidence (caught the cult-classic
        signal even when reception score isn't extreme).
    prestige: dominated by high_reception; popularity_or_award and a
        non-Razzie award bonus together carry the remaining 40%, so
        Best Picture winners outscore equally-reviewed but non-awarded
        peers.
    middle: a soft notability prior — popularity-weighted with reception
        as a tie-breaker. Always active (V1 zeroed this bucket out).
    """
    reception = candidate_row.get("reception_score")
    pop_pct = candidate_row.get("popularity_percentile") or 0.0
    non_razzie = award_signals.non_razzie_score if award_signals else 0.0
    razzie = award_signals.razzie_score if award_signals else 0.0

    if bucket == "cult_garbage":
        low_reception_match = (
            _clamp((50.0 - reception) / 30.0) if reception is not None else 0.0
        )
        pop_match = _clamp((pop_pct - 0.75) / 0.20)
        return _clamp(
            0.40 * low_reception_match + 0.50 * pop_match + 0.10 * razzie
        )

    if bucket == "prestige":
        high_reception_match = (
            _clamp((reception - 75.0) / 20.0) if reception is not None else 0.0
        )
        pop_or_award = max(_clamp((pop_pct - 0.50) / 0.30), non_razzie)
        # Award contribution is additive on top of pop_or_award (clamped
        # via the outer _clamp so a Best Picture winner with strong
        # popularity doesn't exceed 1.0).
        return _clamp(
            0.80 * high_reception_match + 0.20 * pop_or_award + 0.20 * non_razzie
        )

    # middle bucket
    reception_norm = (reception or 0.0) / 100.0
    return _clamp(0.80 * pop_pct + 0.20 * reception_norm)


def _format_score(anchor_bucket: FormatBucket, candidate_row: dict) -> float:
    """Single-anchor format lane: binary same-bucket-or-not."""
    candidate_bucket = format_bucket(candidate_row.get("keyword_ids") or ())
    return 1.0 if candidate_bucket == anchor_bucket else 0.0


def _medium_tags_for_movie(row: dict) -> set[int]:
    """Extract the medium-related keyword IDs from a movie row.

    The medium registry only knows about MEDIUM_TAG_IDS; everything else
    in keyword_ids is irrelevant to medium scoring. Returning a set lets
    callers feed it directly to ``medium_score``.
    """
    return _as_int_set(row.get("keyword_ids")) & MEDIUM_TAG_IDS


def _medium_multiplier(anchor_tags: set[int], candidate_tags: set[int]) -> float:
    """Map the medium similarity score onto a multiplier (V3 §3.2 piecewise).

    V3 splits the multiplier into two regimes:

    - **Cross-category** (live-action ↔ animation): ``medium_score`` returns
      0.0 because MEDIUM_SIMILARITY's LIVE row is all-zeros against animation
      sub-tags (and vice versa). These get a hard ``MEDIUM_CROSS_CATEGORY_MULTIPLIER``
      (0.65) hit — animated Batman shouldn't ride a soft 0.85 floor into a
      live-action Batman anchor's top 10.

    - **Within-category** (any non-zero ``medium_score``): keep the V2 formula
      ``MEDIUM_MULTIPLIER_FLOOR + MEDIUM_MULTIPLIER_RANGE * score``. Perfect
      agreement (1.00) keeps the multiplier at 1.00; CG vs. stop-motion
      (~0.50) lands at ~0.925; same-anim-style at ~0.90 lands at ~0.985.

    Either side missing medium tags is treated as "we can't tell" and
    returns 1.0 (no penalty) — V2 behavior preserved. The harsh
    cross-category penalty only fires when both anchor and candidate
    actually carry medium tags AND those tags don't agree at all.
    """
    if not anchor_tags or not candidate_tags:
        return 1.0
    score = medium_score(anchor_tags, candidate_tags)
    if score == 0.0:
        return MEDIUM_CROSS_CATEGORY_MULTIPLIER
    return MEDIUM_MULTIPLIER_FLOOR + MEDIUM_MULTIPLIER_RANGE * score


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0.0:
        return vector
    return [v / norm for v in vector]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cohesion_weight(avg_pairwise_cosine: float) -> float:
    return _clamp((avg_pairwise_cosine - 0.55) / 0.30, 0.10, 1.0)


def _record_vectors(record: object) -> dict[VectorName, list[float]]:
    raw_vectors = getattr(record, "vector", None)
    if not isinstance(raw_vectors, dict):
        return {}
    out: dict[VectorName, list[float]] = {}
    for vector_name in VectorName:
        raw = raw_vectors.get(vector_name.value)
        if raw is not None:
            out[vector_name] = [float(v) for v in raw]
    return out


class SimilarityQdrantProbeKind(str, Enum):
    """`similarity_qdrant.probe_kind` values — the two Qdrant ops in the flow.

    Mirrors `QdrantProbeKind` on the semantic path: one span name, a `probe_kind`
    attribute discriminating the operations so a reader can filter either.
    """

    ANCHOR_VECTORS = "anchor_vectors"  # _load_anchor_vectors — retrieve stored vecs
    SHAPE = "shape"                    # _query_spaces_batch — batched named-vector probe


async def _load_anchor_vectors(
    anchor_ids: list[int],
) -> dict[int, dict[VectorName, list[float]]]:
    # Manual span around the anchor-vector retrieve: Qdrant speaks gRPC (no
    # auto-instrumentation), so without this the vector-load is an untraced blind
    # spot — the flow's OTHER Qdrant call (the `shape` probe) was the only visible
    # one. `returned_count < requested_count` flags an anchor missing from Qdrant.
    with tracer.start_as_current_span(SIMILARITY_QDRANT) as span:
        span.set_attribute(
            SIMILARITY_QDRANT_PROBE_KIND,
            SimilarityQdrantProbeKind.ANCHOR_VECTORS.value,
        )
        span.set_attribute(
            SIMILARITY_QDRANT_REQUESTED_COUNT, len(anchor_ids)
        )
        records = await qdrant_client.retrieve(
            collection_name=COLLECTION_ALIAS,
            ids=anchor_ids,
            with_payload=False,
            with_vectors=[space.value for space in VectorName],
        )
        span.set_attribute(SIMILARITY_QDRANT_RETURNED_COUNT, len(records))
    return {int(getattr(record, "id")): _record_vectors(record) for record in records}


async def _query_spaces_batch(
    requests: list[tuple[VectorName, list[float], int]],
    *,
    qdrant_filter: Filter | None = None,
) -> list[list[tuple[int, float]]]:
    """Run N named-vector queries against Qdrant in a single round trip.

    Each input tuple is (vector_space, query_vector, per-space limit).
    Output is rows in matching order: ``out[i]`` corresponds to
    ``requests[i]``. This collapses N round trips (the prior
    ``asyncio.gather(*_query_space)`` shape) into one network call —
    Qdrant runs the searches in parallel server-side and shares the
    HNSW page cache across them.

    ``qdrant_filter`` (when not None) is applied to every per-space
    QueryRequest in the batch so all spaces honor the same payload
    filter — typically the user's hard filters translated via
    ``build_qdrant_filter``. None is a no-op (no filter applied).

    Empty input returns ``[]`` so callers can short-circuit when no
    vector spaces are active without a special case here.
    """
    if not requests:
        return []
    query_requests = [
        QueryRequest(
            query=query_vector,
            using=vector_name.value,
            limit=limit,
            filter=qdrant_filter,
            with_payload=False,
            with_vector=False,
            params=QDRANT_SEARCH_PARAMS,
        )
        for vector_name, query_vector, limit in requests
    ]
    # Manual span around the Qdrant batch probe: Qdrant speaks gRPC, which the
    # auto-instrumentation doesn't wrap, so this shape search is otherwise a
    # timing blind spot. Nests under the branch span when run inside the
    # similarity flow; a harmless orphan/no-op otherwise. One call fans out across
    # N named vectors server-side, so the attrs describe the batch (space_count /
    # spaces / per-space + total recall) rather than a single space.
    spaces = [vector_name.value for vector_name, _, _ in requests]
    with tracer.start_as_current_span(SIMILARITY_QDRANT) as span:
        span.set_attribute(
            SIMILARITY_QDRANT_PROBE_KIND,
            SimilarityQdrantProbeKind.SHAPE.value,
        )
        span.set_attribute(SIMILARITY_QDRANT_SPACE_COUNT, len(spaces))
        span.set_attribute(SIMILARITY_QDRANT_SPACES, json.dumps(spaces))
        # Per-space limits are uniform within a call (shape search passes one
        # effective limit across the batch); record it so the 2× over-fetch under
        # an active filter is visible.
        span.set_attribute(
            SIMILARITY_QDRANT_LIMIT_PER_SPACE, requests[0][2]
        )
        span.set_attribute(
            SIMILARITY_QDRANT_FILTER_ACTIVE, qdrant_filter is not None
        )
        responses = await qdrant_client.query_batch_points(
            collection_name=COLLECTION_ALIAS,
            requests=query_requests,
        )
        # Per-space recall (a space starved by the filter shows a short count) plus
        # the total across the batch.
        hits_by_space = {
            space: len(response.points)
            for space, response in zip(spaces, responses, strict=True)
        }
        span.set_attribute(
            SIMILARITY_QDRANT_HITS_BY_SPACE, json.dumps(hits_by_space)
        )
        span.set_attribute(
            SIMILARITY_QDRANT_HIT_COUNT, sum(hits_by_space.values())
        )
    return [
        [(int(point.id), float(point.score)) for point in response.points]
        for response in responses
    ]


def _normalize_space_results(
    rows: list[tuple[int, float]],
    *,
    excluded_ids: set[int],
) -> dict[int, float]:
    raw = {
        movie_id: min(score, 1.0)
        for movie_id, score in rows
        if movie_id not in excluded_ids and score > 0.0
    }
    if not raw:
        return {}
    return normalize_blended_scores(raw)


async def _run_single_anchor_shape_search(
    anchor_id: int,
    anchor_vectors: dict[VectorName, list[float]],
    *,
    qdrant_limit: int,
    metadata_filters: MetadataFilters | None = None,
) -> tuple[dict[int, float], dict[str, float]]:
    active_spaces = {
        space: VECTOR_BASE_WEIGHTS_SINGLE[space]
        for space in VectorName
        if space in anchor_vectors
    }
    space_weights = _normalize_vector_weights(active_spaces)
    if not space_weights:
        return {}, {}

    # Translate the user's hard filters into a Qdrant payload filter once;
    # the same Filter object is reused across every space in the batch.
    # `build_qdrant_filter` returns None when filters are inactive, which
    # is exactly the no-op signal `_query_spaces_batch` expects.
    qdrant_filter = build_qdrant_filter(metadata_filters) if metadata_filters else None

    # Over-fetch when filters are active: HNSW + payload filter satisfies
    # fewer points per traversal step, and the post-filter pool feeds
    # downstream lanes by union, so a thin shape pool weakens shape's
    # contribution to weaving. The multiplier matches the over-fetch
    # discipline elsewhere in the V2 standard flow.
    effective_qdrant_limit = qdrant_limit * 2 if qdrant_filter is not None else qdrant_limit

    # Single batched call instead of N individual queries — one round trip,
    # server-side parallelism, shared HNSW cache warmth across the batch.
    active_space_list = list(space_weights)
    rows_by_space = await _query_spaces_batch(
        [
            (space, anchor_vectors[space], effective_qdrant_limit + 1)
            for space in active_space_list
        ],
        qdrant_filter=qdrant_filter,
    )

    scores: dict[int, float] = {}
    for space, rows in zip(active_space_list, rows_by_space, strict=True):
        normalized = _normalize_space_results(rows, excluded_ids={anchor_id})
        for movie_id, score in normalized.items():
            scores[movie_id] = scores.get(movie_id, 0.0) + space_weights[space] * score

    return _normalize_lane_to_max(scores), {
        space.value: weight for space, weight in space_weights.items()
    }


async def _run_multi_anchor_shape_search(
    anchor_ids: list[int],
    vectors_by_anchor: dict[int, dict[VectorName, list[float]]],
    *,
    qdrant_limit: int,
    metadata_filters: MetadataFilters | None = None,
) -> tuple[dict[int, float], dict[str, float], dict[str, float], float]:
    """Run multi-anchor shape search and return scores + cohesion debug.

    Final tuple element is the mean pairwise cosine across active spaces;
    callers use it to drive the V2 shape-lane scaling and the
    low-cohesion fallback gate.
    """
    usable: dict[VectorName, list[list[float]]] = {}
    for space in VectorName:
        vectors: list[list[float]] = []
        for anchor_id in anchor_ids:
            vector = vectors_by_anchor.get(anchor_id, {}).get(space)
            if vector is None:
                break
            vectors.append(_l2_normalize(vector))
        else:
            usable[space] = vectors

    raw_space_weights: dict[VectorName, float] = {}
    cohesion_debug: dict[str, float] = {}
    avg_pairwise_by_space: dict[VectorName, float] = {}
    centroid_by_space: dict[VectorName, list[float]] = {}

    for space, vectors in usable.items():
        # Pure-Python pairwise dots + element-wise centroid mean on
        # 3072-dim vectors burn through O(N^2 * D) Python ops per space —
        # ~75K multiplies per request for N=3 anchors × 8 spaces. Stack
        # into a single float64 ndarray and use numpy's BLAS-backed
        # matmul + mean instead. Numerical fidelity: IEEE 754 sums are
        # bit-identical for these sizes; any FMA-induced ULP differences
        # are well below the 6-decimal rounding applied in JSON output.
        arr = np.asarray(vectors, dtype=np.float64)  # shape (N_anchors, dim)
        n_anchors = arr.shape[0]
        if n_anchors >= 2:
            # arr @ arr.T is the full Gram matrix of pairwise cosines
            # (vectors are already L2-normalized). The upper triangle
            # excluding the diagonal is the set of unordered pairs.
            gram = arr @ arr.T
            pair_count = n_anchors * (n_anchors - 1) // 2
            avg_pairwise = float(np.triu(gram, k=1).sum()) / pair_count
        else:
            avg_pairwise = 1.0
        avg_pairwise_by_space[space] = avg_pairwise
        # cohesion_weight stays clamped to [0.10, 1.00] for the per-space
        # vector mix — negative weights inside the mix don't make sense.
        # The expanded shape-lane scaling clamp is applied separately by
        # the caller using mean_pairwise_cosine.
        cohesion = _cohesion_weight(avg_pairwise)
        cohesion_debug[space.value] = cohesion
        raw_space_weights[space] = (
            0.75 * cohesion + 0.25 * VECTOR_BASE_WEIGHTS_MULTI[space]
        )

        # Centroid: element-wise mean across anchors, then L2-normalize.
        # Mirrors the prior `[sum(...) / len(...) for idx in range(dim)]`
        # comprehension; numpy does the same arithmetic in one C call.
        centroid_raw = arr.mean(axis=0)
        norm = float(np.sqrt(np.sum(centroid_raw * centroid_raw)))
        if norm <= 0.0:
            centroid_by_space[space] = centroid_raw.tolist()
        else:
            centroid_by_space[space] = (centroid_raw / norm).tolist()

    # Mean pairwise cosine across active spaces — drives V2 shape-lane
    # scaling and the low-cohesion fallback gate.
    if avg_pairwise_by_space:
        mean_pairwise_cosine = sum(avg_pairwise_by_space.values()) / len(
            avg_pairwise_by_space
        )
    else:
        mean_pairwise_cosine = 0.0

    space_weights = _normalize_vector_weights(raw_space_weights)
    if not space_weights:
        return {}, {}, cohesion_debug, mean_pairwise_cosine

    # Same hard-filter translation as the single-anchor path (one Filter
    # object shared across every space in the batch).
    qdrant_filter = build_qdrant_filter(metadata_filters) if metadata_filters else None

    # Same over-fetch rationale as single-anchor: filtered HNSW returns
    # fewer points per step, and the downstream lanes consume this pool
    # by union, so a thin filtered slice weakens shape's contribution.
    effective_qdrant_limit = qdrant_limit * 2 if qdrant_filter is not None else qdrant_limit

    # Single batched call across active spaces; see _query_spaces_batch.
    active_space_list = list(space_weights)
    per_space_limit = effective_qdrant_limit + len(anchor_ids)
    rows_by_space = await _query_spaces_batch(
        [
            (space, centroid_by_space[space], per_space_limit)
            for space in active_space_list
        ],
        qdrant_filter=qdrant_filter,
    )

    excluded = set(anchor_ids)
    scores: dict[int, float] = {}
    for space, rows in zip(active_space_list, rows_by_space, strict=True):
        normalized = _normalize_space_results(rows, excluded_ids=excluded)
        for movie_id, score in normalized.items():
            scores[movie_id] = scores.get(movie_id, 0.0) + space_weights[space] * score

    return (
        _normalize_lane_to_max(scores),
        {space.value: weight for space, weight in space_weights.items()},
        cohesion_debug,
        mean_pairwise_cosine,
    )


def _shape_raw_for_multi_anchor(mean_pairwise_cosine: float) -> float:
    """V2: scale shape-lane raw weight by overall vector cohesion.

    Maps mean_pairwise_cosine through clamp((x - 0.55)/0.30, -0.40, 1.00)
    onto a [-0.40, 1.00] signal. shape_raw = 0.60 * (1 + signal) ranges
    over [0.36, 1.20] — boosting shape on coherent sets (LotR trilogy
    sits near 1.20) and penalizing it on incoherent ones (chaotic mixed
    anchor sets sit near 0.36).
    """
    signal = _clamp((mean_pairwise_cosine - 0.55) / 0.30, -0.40, 1.00)
    return 0.60 * (1.0 + signal)


def _build_results(
    *,
    anchor_ids: list[int],
    lane_scores: dict[LaneName, dict[int, float]],
    lane_weights: dict[LaneName, float],
    limit: int,
    # Post-additive multipliers — each is a per-movie lookup. None means
    # "this signal isn't relevant for this flow" (e.g., country/language is
    # multi-anchor only; medium tags may be empty for live-action anchors).
    studio_score_by_movie: dict[int, float] | None = None,
    medium_multiplier_by_movie: dict[int, float] | None = None,
    country_consensus_match_by_movie: dict[int, bool] | None = None,
    # V3 floor signals (each gated on shape ≥ floor's shape gate).
    # V3.4.4: rare_keyword floor removed — distinctive low-shape match
    # exploration runs through the rare_keyword bucket in the weaver.
    cast_floor_by_movie: dict[int, float] | None = None,
    director_floor_max_ratio_by_movie: dict[int, float] | None = None,
    # V3.1: per-flow director floor params. Defaults preserve V3
    # multi-anchor behavior; single-anchor callers pass softer values
    # (shape gate 0.20, ratio threshold 1.0) so a curated-auteur match
    # with shape ≥ 0.20 floors at 0.35.
    director_floor_ratio_threshold: float = DIRECTOR_FLOOR_RATIO_THRESHOLD,
    director_floor_shape_gate: float = DIRECTOR_FLOOR_SHAPE_GATE,
    director_floor_magnitude: float = DIRECTOR_FLOOR_MAGNITUDE,
    # V3 §3.1 shorts handling.
    candidate_format_bucket_by_movie: dict[int, FormatBucket] | None = None,
    anchor_active_format_bucket: FormatBucket | None = None,
    apply_shorts_downrank: bool = True,
    # V3.3 shape multiplier — anchor cohort's per-shape cohesion AND
    # per-candidate shape classification. When the candidate's shape
    # matches a shape the cohort has cohesion ≥0.5 in, apply
    # ×(1 + max_strength * cohesion). Defaults preserve no-shape-boost
    # behavior.
    anchor_shape_cohesion: dict[str, float] | None = None,
    candidate_shape_by_movie: dict[int, str | None] | None = None,
    # When the multi-anchor anchor set is shorts-dominant (≥50% of
    # anchors are shorts), the short candidates get a small positive
    # boost via SHORTS_MULTI_ANCHOR_BOOST. Mutually exclusive with the
    # downrank — callers pass `apply_shorts_downrank=False,
    # apply_shorts_boost=True` together.
    apply_shorts_boost: bool = False,
    anchor_format_bucket: FormatBucket | None = None,
    enforce_format_top_lock: bool = False,
    # V3.4 bucket-weaver inputs. When omitted, the weaver collapses to
    # best-overall-only (equivalent to no weaving past format lock).
    bucket_signals: dict[str, float] | None = None,
    bucket_memberships_by_movie: dict[int, set[str]] | None = None,
    # V3.4.6 franchise fatigue (single-anchor flow only). When
    # `enforce_franchise_fatigue=True`, the weaver bans franchise
    # candidates whenever the placed franchise:non-franchise ratio
    # exceeds FRANCHISE_FATIGUE_THRESHOLD. `is_franchise_by_movie`
    # flags each candidate; multi-anchor leaves both at default.
    enforce_franchise_fatigue: bool = False,
    is_franchise_by_movie: dict[int, bool] | None = None,
    # V3.4.9 director gap-boost (single-anchor flow only). When set,
    # the weaver tracks consecutive non-director-match placements and
    # applies a compounding (1 + DIRECTOR_GAP_INCREMENT) ** k multiplier
    # to every director-match candidate's effective score for MMR
    # comparison once the gap reaches DIRECTOR_GAP_THRESHOLD. Multi-
    # anchor leaves this off — same-director match in multi-anchor is
    # cohesion-scaled, not a single-auteur signal.
    enforce_director_gap_boost: bool = False,
) -> tuple[list[SimilarMovieResult], dict[LaneName, int]]:
    """Combine per-lane scores into ranked candidates with V3 multipliers + floors.

    Pipeline:
      1. Sum lane_weights * lane_scores across ADDITIVE_LANES (director
         is a passthrough lane — its raw score IS the absolute
         contribution).
      2. Apply post-additive multipliers (studio, V3.4.3 rare-keyword,
         country/language, medium piecewise, format-mismatch, shorts
         harsh downrank) on shape-qualifying candidates.
      3. Apply V3 floors (rare-keyword, cast bucket-with-floor, director
         cohesion floor) gated by shape — these allow a candidate with
         strong distinctive-match evidence to override moderate shape
         drift, but never with weak shape.
      4. Weave: V1 dominance/franchise caps plus the V2 top-5 format
         lock plus the V3 max-1 shorts cap (when anchor isn't a short).

    Studio is debug-only in lane_scores — its multiplier path is the
    studio_score_by_movie dict, not a contribution to the additive sum.
    """
    excluded = set(anchor_ids)
    candidate_ids: set[int] = set()
    for scores in lane_scores.values():
        candidate_ids.update(scores)
    candidate_ids -= excluded

    studio_score_by_movie = studio_score_by_movie or {}
    medium_multiplier_by_movie = medium_multiplier_by_movie or {}
    country_consensus_match_by_movie = country_consensus_match_by_movie or {}
    cast_floor_by_movie = cast_floor_by_movie or {}
    director_floor_max_ratio_by_movie = director_floor_max_ratio_by_movie or {}
    candidate_format_bucket_by_movie = candidate_format_bucket_by_movie or {}
    anchor_shape_cohesion = anchor_shape_cohesion or {}
    candidate_shape_by_movie = candidate_shape_by_movie or {}

    # Whether the shorts harsh downrank fires for this flow. Disabled
    # when the anchor IS a short (or multi-anchor with shorts dominance);
    # callers pass apply_shorts_downrank=False in those cases.
    apply_shorts = (
        apply_shorts_downrank
        and anchor_active_format_bucket != "short"
    )

    # =====================================================================
    # Vectorized scoring — the per-candidate scalar arithmetic below was
    # the longest pure-Python phase of the similarity flow (10 dict-of-
    # dict lookups + weighted sum + 6 conditional multipliers per row,
    # over thousands of candidates). Build numpy column arrays for the
    # raw lane scores and every per-movie auxiliary signal, do the
    # weighted sum + multipliers + floors as elementwise ndarray ops in
    # C/BLAS, then iterate ONCE over the indexed arrays to materialize
    # the diagnostic-bearing _CandidateScore objects. Scoring semantics
    # are preserved bit-for-bit modulo BLAS FMA ULP differences (well
    # below the 6-decimal rounding applied in JSON output).
    # =====================================================================

    candidates: list[_CandidateScore] = []
    if not candidate_ids:
        counts: dict[LaneName, int] = {
            lane: len(scores) for lane, scores in lane_scores.items()
        }
        if studio_score_by_movie:
            counts["studio"] = sum(
                1 for v in studio_score_by_movie.values() if v > 0.0
            )
        return [], counts

    # ---- Index layout -----------------------------------------------------
    candidate_ids_list: list[int] = list(candidate_ids)
    n_candidates = len(candidate_ids_list)
    id_to_idx = {mid: idx for idx, mid in enumerate(candidate_ids_list)}
    lane_idx_map = {lane: idx for idx, lane in enumerate(ALL_LANES)}
    shape_col = lane_idx_map["shape"]
    director_col = lane_idx_map["director"]
    additive_idx_in_all = [lane_idx_map[lane] for lane in ADDITIVE_LANES]

    # ---- score_matrix: (n_candidates, len(ALL_LANES)) ---------------------
    # Populate per-lane sparsely (most lanes only score a fraction of the
    # candidate pool). lane_scores["studio"] is empty by contract — the
    # studio_col stays at 0 throughout the additive sum and is later
    # overwritten in the diagnostic per_lane dict with studio_score_arr.
    score_matrix = np.zeros((n_candidates, len(ALL_LANES)), dtype=np.float64)
    for lane, lane_dict in lane_scores.items():
        lane_idx = lane_idx_map.get(lane)
        if lane_idx is None or not lane_dict:
            continue
        for mid, val in lane_dict.items():
            idx = id_to_idx.get(mid)
            if idx is not None:
                score_matrix[idx, lane_idx] = val
    np.clip(score_matrix, 0.0, 1.0, out=score_matrix)

    shape_arr = score_matrix[:, shape_col]
    director_lane_arr = score_matrix[:, director_col]

    # ---- Auxiliary per-movie arrays ---------------------------------------
    studio_score_arr = np.zeros(n_candidates, dtype=np.float64)
    for mid, val in studio_score_by_movie.items():
        idx = id_to_idx.get(mid)
        if idx is not None:
            studio_score_arr[idx] = val

    medium_mult_arr = np.ones(n_candidates, dtype=np.float64)
    for mid, val in medium_multiplier_by_movie.items():
        idx = id_to_idx.get(mid)
        if idx is not None:
            medium_mult_arr[idx] = val

    # Country: tristate per candidate — match (1) / mismatch (0) /
    # absent (-1). Mirrors the original's `if movie_id in dict` guard.
    country_state_arr = np.full(n_candidates, -1, dtype=np.int8)
    for mid, matched in country_consensus_match_by_movie.items():
        idx = id_to_idx.get(mid)
        if idx is not None:
            country_state_arr[idx] = 1 if matched else 0

    cast_floor_arr = np.zeros(n_candidates, dtype=np.float64)
    for mid, val in cast_floor_by_movie.items():
        idx = id_to_idx.get(mid)
        if idx is not None:
            cast_floor_arr[idx] = val

    director_ratio_arr = np.zeros(n_candidates, dtype=np.float64)
    for mid, val in director_floor_max_ratio_by_movie.items():
        idx = id_to_idx.get(mid)
        if idx is not None:
            director_ratio_arr[idx] = val

    # Format bucket — keep as Python list so we can compare to FormatBucket
    # literals; not amenable to ndarray vector ops.
    cand_fmt_list: list[FormatBucket | None] = [None] * n_candidates
    for mid, fmt in candidate_format_bucket_by_movie.items():
        idx = id_to_idx.get(mid)
        if idx is not None:
            cand_fmt_list[idx] = fmt

    # Per-candidate shape multiplier. Memoize on the candidate's shape
    # label so we run `_shape_multiplier` at most once per distinct shape.
    shape_mult_arr = np.ones(n_candidates, dtype=np.float64)
    shape_mult_cache: dict[str | None, float] = {}
    for mid, cand_shape in candidate_shape_by_movie.items():
        idx = id_to_idx.get(mid)
        if idx is None:
            continue
        cached = shape_mult_cache.get(cand_shape)
        if cached is None:
            cached = _shape_multiplier(anchor_shape_cohesion, cand_shape)
            shape_mult_cache[cand_shape] = cached
        shape_mult_arr[idx] = cached

    # ---- Skip mask: candidates with no nonzero lane (matches original
    # `if not sources: continue`). Studio is multiplier-only and lives at
    # studio_col with all zeros, so it doesn't lift any candidate over
    # the threshold here — same behavior as the original loop, where
    # studio's contribution to `sources` happened only via the later
    # append (after the early-continue check).
    has_any_lane = (score_matrix > 0.0).any(axis=1)

    # ---- Additive base score ---------------------------------------------
    weights_vec = np.array(
        [lane_weights.get(lane, 0.0) for lane in ALL_LANES],
        dtype=np.float64,
    )
    contribution_matrix = score_matrix * weights_vec  # (N, L), debug-readable
    base_scores = contribution_matrix.sum(axis=1)     # equivalent: weights_vec @ score_matrix.T

    # ---- Dominant lane (over ADDITIVE_LANES, tiebreak by raw, ties to
    # first ADDITIVE_LANES order). Iterating the 8 additive lanes and
    # carrying running best matches Python's `max(..., key=lambda)`
    # semantics exactly — including "first wins on full tie".
    additive_contribs = contribution_matrix[:, additive_idx_in_all]
    additive_raws = score_matrix[:, additive_idx_in_all]
    best_contrib = additive_contribs[:, 0].copy()
    best_raw = additive_raws[:, 0].copy()
    best_dom_idx = np.zeros(n_candidates, dtype=np.intp)
    for k in range(1, len(ADDITIVE_LANES)):
        contrib_k = additive_contribs[:, k]
        raw_k = additive_raws[:, k]
        # Lex order: (contrib, raw). Strict > on contrib, OR equal contrib
        # AND strict > on raw. Match Python's max key semantics.
        replace = (contrib_k > best_contrib) | (
            (contrib_k == best_contrib) & (raw_k > best_raw)
        )
        best_contrib = np.where(replace, contrib_k, best_contrib)
        best_raw = np.where(replace, raw_k, best_raw)
        best_dom_idx = np.where(replace, k, best_dom_idx)

    # ---- Multiplier masks + arrays ---------------------------------------
    # Studio multiplier
    studio_mask = (studio_score_arr > 0.0) & (shape_arr >= STUDIO_MULTIPLIER_SHAPE_GATE)
    studio_mult_arr = np.where(
        studio_mask,
        1.0 + STUDIO_MULTIPLIER_STRENGTH * studio_score_arr,
        1.0,
    )

    # Director multiplier
    director_mask = (director_lane_arr > 0.0) & (
        shape_arr >= DIRECTOR_MULTIPLIER_SHAPE_GATE
    )
    director_mult_arr = np.where(
        director_mask,
        1.0 + DIRECTOR_MULTIPLIER_STRENGTH * director_lane_arr,
        1.0,
    )

    # Country multiplier — tristate. Match → BOOST; mismatch → PENALTY;
    # absent → 1.0 (no contribution).
    country_present_mask = country_state_arr >= 0
    country_match_mask = country_state_arr == 1
    country_mult_arr = np.where(
        country_present_mask,
        np.where(country_match_mask, COUNTRY_CONSENSUS_BOOST, COUNTRY_CONSENSUS_PENALTY),
        1.0,
    )

    # Format-mismatch multiplier — needs Python iteration over the
    # FormatBucket list since shorts and None get special handling.
    fmt_mismatch_arr = np.ones(n_candidates, dtype=np.float64)
    if anchor_format_bucket is not None:
        for idx, cand_fmt in enumerate(cand_fmt_list):
            if (
                cand_fmt is not None
                and cand_fmt != anchor_format_bucket
                and cand_fmt != "short"
            ):
                fmt_mismatch_arr[idx] = FORMAT_CROSS_CATEGORY_MULTIPLIER

    # Shorts downrank / boost — exclusive: a candidate at most lands in
    # one or the other. Apply via a single multiplier array with parallel
    # boolean masks for diagnostics.
    shorts_mult_arr = np.ones(n_candidates, dtype=np.float64)
    shorts_downrank_mask = np.zeros(n_candidates, dtype=bool)
    shorts_boost_mask = np.zeros(n_candidates, dtype=bool)
    if apply_shorts or apply_shorts_boost:
        for idx, cand_fmt in enumerate(cand_fmt_list):
            if cand_fmt != "short":
                continue
            if apply_shorts:
                shorts_mult_arr[idx] = SHORTS_DOWNRANK_MULTIPLIER
                shorts_downrank_mask[idx] = True
            elif apply_shorts_boost:
                shorts_mult_arr[idx] = SHORTS_MULTI_ANCHOR_BOOST
                shorts_boost_mask[idx] = True

    # ---- Apply all multipliers (left-to-right matches original order) -----
    scores_arr = (
        base_scores
        * studio_mult_arr
        * director_mult_arr
        * country_mult_arr
        * medium_mult_arr
        * fmt_mismatch_arr
        * shorts_mult_arr
        * shape_mult_arr
    )

    # ---- Floors -----------------------------------------------------------
    cast_floor_active = (cast_floor_arr > 0.0) & (shape_arr >= CAST_FLOOR_SHAPE_GATE)
    cast_floor_effective = np.where(cast_floor_active, cast_floor_arr, 0.0)
    director_floor_active = (
        (director_ratio_arr >= director_floor_ratio_threshold)
        & (shape_arr >= director_floor_shape_gate)
    )
    director_floor_effective = np.where(
        director_floor_active, director_floor_magnitude, 0.0
    )
    # max() on a list of (label, value) ties → returns FIRST. The
    # original appends "cast" before "director", so cast wins on tie.
    best_floor_value = np.maximum(cast_floor_effective, director_floor_effective)
    cast_wins = cast_floor_active & (cast_floor_effective >= director_floor_effective)
    director_wins = director_floor_active & ~cast_wins
    # Floor displaces score only on STRICT > to match the original semantics.
    floor_displaces = best_floor_value > scores_arr
    final_scores_arr = np.where(floor_displaces, best_floor_value, scores_arr)
    applied_floor_value_arr = np.where(floor_displaces, best_floor_value, 0.0)

    # ---- Per-candidate object construction (single Python pass) -----------
    additive_lane_names = list(ADDITIVE_LANES)
    # Cached clamped studio for the diagnostic per_lane["studio"] field.
    studio_diag_arr = np.clip(studio_score_arr, 0.0, 1.0)

    for idx in range(n_candidates):
        if not has_any_lane[idx]:
            continue
        mid = candidate_ids_list[idx]

        row = score_matrix[idx]
        # per_lane mirrors the original dict — ALL_LANES keys with the
        # clamped raw score. Studio gets overwritten with its multiplier-
        # path value (preserves the diagnostic-only studio appearance).
        per_lane = {lane: float(row[lane_idx_map[lane]]) for lane in ALL_LANES}
        per_lane["studio"] = float(studio_diag_arr[idx])

        # Sources list: ALL_LANES order excluding studio (which is
        # multiplier-only and appended last when its score is positive).
        # This matches the original construction precisely.
        sources_list: list[LaneName] = [
            lane
            for lane in ALL_LANES
            if lane != "studio" and per_lane[lane] > 0.0
        ]
        if studio_score_arr[idx] > 0.0:
            sources_list.append("studio")

        dominant = additive_lane_names[int(best_dom_idx[idx])]

        applied_multipliers: dict[str, float] = {}
        if studio_mask[idx]:
            applied_multipliers["studio"] = float(studio_mult_arr[idx])
        if director_mask[idx]:
            applied_multipliers["director"] = float(director_mult_arr[idx])
        if country_present_mask[idx]:
            applied_multipliers["country"] = float(country_mult_arr[idx])
        if medium_mult_arr[idx] != 1.0:
            applied_multipliers["medium"] = float(medium_mult_arr[idx])
        if fmt_mismatch_arr[idx] != 1.0:
            applied_multipliers["format_mismatch"] = float(fmt_mismatch_arr[idx])
        if shorts_downrank_mask[idx]:
            applied_multipliers["shorts_downrank"] = SHORTS_DOWNRANK_MULTIPLIER
        elif shorts_boost_mask[idx]:
            applied_multipliers["shorts_boost"] = SHORTS_MULTI_ANCHOR_BOOST
        if shape_mult_arr[idx] != 1.0:
            applied_multipliers["shape"] = float(shape_mult_arr[idx])

        if floor_displaces[idx]:
            if cast_wins[idx]:
                applied_floor_source = "cast"
            elif director_wins[idx]:
                applied_floor_source = "director"
            else:
                applied_floor_source = ""
        else:
            applied_floor_source = ""

        candidates.append(
            _CandidateScore(
                movie_id=mid,
                score=float(final_scores_arr[idx]),
                lane_scores=per_lane,
                candidate_sources=sources_list,
                dominant_lane=dominant,
                base_score=float(base_scores[idx]),
                multipliers=applied_multipliers,
                floor_value=float(applied_floor_value_arr[idx]),
                floor_source=applied_floor_source,
                director_match=bool(director_lane_arr[idx] > 0.0),
            )
        )

    woven, weave_targets = _weave_candidates(
        candidates,
        limit=limit,
        bucket_signals=bucket_signals,
        bucket_memberships_by_movie=bucket_memberships_by_movie,
        anchor_format_bucket=anchor_format_bucket,
        enforce_format_top_lock=enforce_format_top_lock,
        candidate_format_bucket_by_movie=candidate_format_bucket_by_movie,
        enforce_shorts_cap=apply_shorts,
        enforce_franchise_fatigue=enforce_franchise_fatigue,
        is_franchise_by_movie=is_franchise_by_movie,
        enforce_director_gap_boost=enforce_director_gap_boost,
    )
    # Surface the weaver's DESIRED per-bucket allocation onto the branch span (the
    # current span when this runs inside the similarity entity flow; a no-op
    # otherwise). This is the target ratio `_compute_bucket_targets` set before
    # weaving — "how many auteur / franchise / lead-actor rows the list was meant
    # to reserve" — not the realized draw (which multi-bucket credit routes through
    # best_overall). See `_record_weave_targets` for the seats-vs-targets rationale.
    _record_weave_targets(weave_targets)
    ranked = [
        SimilarMovieResult(
            movie_id=c.movie_id,
            score=c.score,
            evidence=LaneEvidence(
                lane_scores=c.lane_scores,
                candidate_sources=c.candidate_sources,
                dominant_lane=c.dominant_lane,
                base_score=c.base_score,
                multipliers=dict(c.multipliers),
                floor_value=c.floor_value,
                floor_source=c.floor_source,
            ),
        )
        for c in woven
    ]
    # Debug counts cover every lane that contributed at least one candidate
    # — including studio, which doesn't appear in lane_scores when it's
    # multiplier-only, so synthesize from studio_score_by_movie.
    counts: dict[LaneName, int] = {
        lane: len(scores) for lane, scores in lane_scores.items()
    }
    if studio_score_by_movie:
        counts["studio"] = sum(
            1 for v in studio_score_by_movie.values() if v > 0.0
        )
    return ranked, counts


def _base_sort_key(candidate: _CandidateScore) -> tuple[float, int, int]:
    return (-candidate.score, -len(candidate.candidate_sources), candidate.movie_id)


def _can_enter_top_section(
    candidate: _CandidateScore,
    *,
    best_score: float,
    dominant_counts: dict[LaneName, int],
    franchise_count: int,
    top_format_lock_active: bool,
) -> bool:
    """Decide whether a candidate may enter the top section of the page.

    V2 changes vs. V1: the studio dominance / weak-studio-source guard is
    gone (studio is no longer dominant in any candidate, since it's
    multiplier-only). The format top-5 lock is enforced via the
    `top_format_lock_active` flag — candidates that don't share the
    anchor's format bucket are deferred while we're filling those first
    slots.
    """
    dominant_lane = candidate.dominant_lane
    shape_score = candidate.lane_scores["shape"]

    if top_format_lock_active and candidate.lane_scores["format"] < 1.0:
        return False
    if dominant_counts.get(dominant_lane, 0) >= MAX_TOP_DOMINANT_LANE:
        return False
    if candidate.lane_scores["franchise"] > 0.0 and franchise_count >= MAX_TOP_FRANCHISE:
        return False
    if dominant_lane != "shape" and candidate.score < best_score - COMPETITIVE_BAND:
        return False
    if dominant_lane == "source" and shape_score < 0.45:
        return False
    if dominant_lane == "franchise" and shape_score < 0.35:
        return False
    if dominant_lane == "director" and shape_score < 0.40:
        return False
    return True


def _compute_bucket_targets(
    bucket_signals: dict[str, float],
    *,
    total_slots: int,
) -> dict[str, int]:
    """V3.4: derive per-bucket slot targets from anchor signal scores.

    Best-overall always gets the floor (5/10) plus any slack from
    capped or un-instantiated signal buckets. Signal buckets share the
    remaining 5 slots proportional to their signal strength, capped
    individually at WEAVER_BUCKET_CAP (3).

    Buckets with `signal < WEAVER_BUCKET_INSTANTIATE_MIN` (0.30) are
    not instantiated — their slots flow back to best-overall. This
    is what makes the weaver respect "if the auteur isn't notable,
    don't include an auteur row".
    """
    target = {BUCKET_BEST_OVERALL: WEAVER_BEST_OVERALL_FLOOR}
    remaining = total_slots - WEAVER_BEST_OVERALL_FLOOR
    if remaining <= 0:
        target[BUCKET_BEST_OVERALL] = total_slots
        return target

    instantiated = [
        b for b, sig in bucket_signals.items()
        if b != BUCKET_BEST_OVERALL and sig >= WEAVER_BUCKET_INSTANTIATE_MIN
    ]
    if not instantiated:
        target[BUCKET_BEST_OVERALL] = total_slots
        return target

    total_signal = sum(bucket_signals[b] for b in instantiated)
    if total_signal <= 0.0:
        target[BUCKET_BEST_OVERALL] = total_slots
        return target

    # Hamilton's largest-remainder method — distributes integer slots
    # proportional to signal strength while guaranteeing the sum stays
    # within `remaining`. Plain `round()` per-bucket can overshoot when
    # all three signal buckets sit at equal strength (e.g. 3 × round(1.667)
    # = 6 > 5 remaining slots), which would steal from best-overall's
    # floor; the largest-remainder pass distributes the leftover slots
    # explicitly and respects WEAVER_BUCKET_CAP.
    shares = {
        b: remaining * bucket_signals[b] / total_signal for b in instantiated
    }
    floored = {b: min(WEAVER_BUCKET_CAP, int(shares[b])) for b in instantiated}
    leftover = remaining - sum(floored.values())
    if leftover > 0:
        remainders = sorted(
            instantiated,
            key=lambda b: -(shares[b] - int(shares[b])),
        )
        for b in remainders:
            if leftover <= 0:
                break
            if floored[b] < WEAVER_BUCKET_CAP:
                floored[b] += 1
                leftover -= 1

    for b in instantiated:
        target[b] = floored[b]
    # Any remaining slack (from per-bucket cap saturation) flows back
    # to best-overall, preserving the always-on bucket invariant.
    slack = remaining - sum(floored.values())
    target[BUCKET_BEST_OVERALL] += max(0, slack)
    return target


def _peek_next_eligible_for_bucket(
    queue: list[_CandidateScore],
    queue_pos: int,
    used_movie_ids: set[int],
    *,
    slot_index: int,
    anchor_format_bucket: FormatBucket | None,
    enforce_format_top_lock: bool,
    candidate_format_bucket_by_movie: dict[int, FormatBucket],
    enforce_shorts_cap: bool,
    shorts_count: int,
    enforce_franchise_fatigue: bool = False,
    is_franchise_by_movie: dict[int, bool] | None = None,
    franchise_count: int = 0,
    non_franchise_count: int = 0,
) -> tuple[_CandidateScore, int] | None:
    """Walk a bucket's V3-rank queue from ``queue_pos`` to find the next
    pickable candidate — one not already placed, satisfying the format
    top-5 lock at slot indices < TOP_FORMAT_LOCK, and the shorts cap.

    When ``enforce_franchise_fatigue`` is set (single-anchor flow),
    franchise candidates are skipped while
    ``franchise_count > FRANCHISE_FATIGUE_THRESHOLD * non_franchise_count``.
    The fatigue gate forces the weaver to fall through to the next
    non-franchise pick from the same or any other bucket.

    Returns ``(candidate, new_pos)`` where ``new_pos`` is the index AT
    the candidate (not past it). Caller advances past it on placement.
    """
    is_franchise_by_movie = is_franchise_by_movie or {}
    franchise_fatigue_active = (
        enforce_franchise_fatigue
        and franchise_count > FRANCHISE_FATIGUE_THRESHOLD * non_franchise_count
    )
    pos = queue_pos
    while pos < len(queue):
        c = queue[pos]
        if c.movie_id in used_movie_ids:
            pos += 1
            continue
        # Format top-5 lock: while filling slots 0..TOP_FORMAT_LOCK-1
        # with `enforce_format_top_lock` set, only same-format candidates
        # may be placed. From slot TOP_FORMAT_LOCK onward, format lock
        # disengages and any candidate may be placed.
        if (
            enforce_format_top_lock
            and anchor_format_bucket is not None
            and slot_index < TOP_FORMAT_LOCK
            and c.lane_scores.get("format", 0.0) < 1.0
        ):
            pos += 1
            continue
        # Shorts cap: at most SHORTS_TOP_SECTION_MAX (1) short in the
        # top section when the anchor isn't a short. Implemented by
        # skipping shorts once the cap is reached.
        if (
            enforce_shorts_cap
            and shorts_count >= SHORTS_TOP_SECTION_MAX
            and candidate_format_bucket_by_movie.get(c.movie_id) == "short"
        ):
            pos += 1
            continue
        # V3.4.6 franchise fatigue (single-anchor only): once the placed
        # ratio of franchise to non-franchise exceeds the threshold,
        # ban franchise candidates regardless of which bucket is asking.
        # The weaver re-evaluates fatigue between slots, so the gate
        # naturally lifts as more non-franchise picks accumulate.
        if franchise_fatigue_active and is_franchise_by_movie.get(c.movie_id, False):
            pos += 1
            continue
        return c, pos
    return None


def _weave_candidates(
    candidates: list[_CandidateScore],
    *,
    limit: int,
    bucket_signals: dict[str, float] | None = None,
    bucket_memberships_by_movie: dict[int, set[str]] | None = None,
    anchor_format_bucket: FormatBucket | None = None,
    enforce_format_top_lock: bool = False,
    candidate_format_bucket_by_movie: dict[int, FormatBucket] | None = None,
    enforce_shorts_cap: bool = False,
    # V3.4.6 franchise fatigue (single-anchor flows pass these in).
    enforce_franchise_fatigue: bool = False,
    is_franchise_by_movie: dict[int, bool] | None = None,
    # V3.4.9 director gap-boost (single-anchor flow only).
    enforce_director_gap_boost: bool = False,
) -> tuple[list[_CandidateScore], dict[str, int]]:
    """V3.4 bucket-weaver: greedy slot-by-slot fill with MMR-style
    starvation boost.

    Replaces the V3 dominance/franchise/competitive-band caps in
    ``_can_enter_top_section`` with explicit per-bucket targets. When
    ``bucket_signals`` is empty or no bucket clears
    ``WEAVER_BUCKET_INSTANTIATE_MIN``, the algorithm collapses to "all
    10 slots from best_overall in V3-rank order" — equivalent to no
    weaving past the format lock + shorts cap.

    Algorithm per slot 1..min(limit, TOP_SECTION_SIZE):
    1. For each bucket whose target isn't met, peek the top unplaced
       format-eligible candidate from its V3-rank queue.
    2. Compute adjusted_score = (1−λ)*relevance + λ*deficit_ratio.
    3. Pick the bucket with the highest adjusted score; place its
       candidate; bump placed[] for every bucket the candidate is a
       member of (full credit on placement).

    Past slot TOP_SECTION_SIZE, append remaining unplaced candidates in
    V3-rank order so the API can still return up to ``limit`` items.

    Returns ``(woven, target)`` where ``target`` is the DESIRED per-bucket slot
    allocation computed up front by ``_compute_bucket_targets`` (best_overall's
    floor plus any signal bucket that cleared the instantiation threshold). This
    is the intended ratio surfaced as similarity observability — deliberately the
    target, not the realized draw: multi-bucket full credit (below) means a film
    that belongs to a signal bucket but is picked via best_overall counts toward
    that bucket's quota without the bucket ever winning a slot, so a per-slot draw
    tally reads as "best_overall took everything" for franchise-dominant cohorts.
    The target answers the more useful question — what the weave meant to reserve.
    """
    if not candidates or limit <= 0:
        return [], {}

    bucket_signals = bucket_signals or {}
    bucket_memberships_by_movie = bucket_memberships_by_movie or {}
    candidate_format_bucket_by_movie = candidate_format_bucket_by_movie or {}

    base_sorted = sorted(candidates, key=_base_sort_key)
    top_section_cap = min(TOP_SECTION_SIZE, limit)
    if not base_sorted:
        return [], {}

    # Targets: best_overall is always present; signal-buckets only when
    # they clear the instantiation threshold.
    target = _compute_bucket_targets(bucket_signals, total_slots=top_section_cap)

    # Per-bucket queues — V3-rank order. best_overall is the universal
    # queue (every candidate is a member). Other buckets only include
    # their qualifying candidates.
    queue_by_bucket: dict[str, list[_CandidateScore]] = {b: [] for b in target}
    for c in base_sorted:
        memberships = bucket_memberships_by_movie.get(c.movie_id, set()) | {
            BUCKET_BEST_OVERALL
        }
        for b in memberships:
            if b in target:
                queue_by_bucket[b].append(c)

    # MMR relevance is the V3 score normalized by the global top score
    # (so best_overall's top candidate ≈ 1.0 and signal-bucket tops are
    # typically < 1.0). Avoid division by zero on degenerate cohorts.
    best_score = base_sorted[0].score
    if best_score <= 0.0:
        best_score = 1.0

    placed: dict[str, int] = {b: 0 for b in target}
    queue_pos: dict[str, int] = {b: 0 for b in target}
    used_movie_ids: set[int] = set()
    shorts_count = 0
    # V3.4.6 franchise fatigue tracker (single-anchor). Counts placed
    # candidates partitioned by `is_franchise_by_movie` membership.
    is_franchise_by_movie = is_franchise_by_movie or {}
    franchise_count = 0
    non_franchise_count = 0
    # V3.4.9 director gap-boost tracker. Counts consecutive non-director
    # placements; resets to 0 when a director-match film is placed.
    # Used to compute extra multiplier strength applied to director
    # candidates' effective scores during MMR comparison.
    consecutive_non_director = 0
    woven: list[_CandidateScore] = []

    for slot_index in range(top_section_cap):
        # V3.4.9 gap-boost: compounding rubber-band on the full score of
        # any director-match candidate once we've gone
        # ``DIRECTOR_GAP_THRESHOLD`` slots without placing one. Exponent
        # k = max(0, gap - THRESHOLD + 1) so the boost engages AT the
        # threshold (gap=3 → k=1 → ×1.10), then compounds (gap=4 → ×1.21,
        # gap=5 → ×1.331, …). Resets the moment a director-match film is
        # placed. Single-anchor only — multi-anchor leaves the flag off.
        if enforce_director_gap_boost and consecutive_non_director >= DIRECTOR_GAP_THRESHOLD:
            gap_boost_exponent = consecutive_non_director - DIRECTOR_GAP_THRESHOLD + 1
            gap_boost_multiplier = (1.0 + DIRECTOR_GAP_INCREMENT) ** gap_boost_exponent
        else:
            gap_boost_multiplier = 1.0
        # For each bucket that still has a deficit, peek the top unplaced
        # format-eligible candidate. Compute the MMR-adjusted score.
        # Best-overall is the always-on bucket — it competes on every
        # slot regardless of placed[] count. Multi-bucket full credit
        # can drive placed[best_overall] past target via cross-bucket
        # placements (a Toy Story 3 picked from best_overall also
        # increments rare_keyword and lead_actor); without this carve-
        # out, best_overall would gate out by slot 5 and the algorithm
        # would short-circuit before filling the 10-slot top section.
        eligible: list[tuple[str, _CandidateScore, float, int, float]] = []
        # V3.4.9: when the gap-boost is active, keep the auteur bucket
        # peek-able past its placed quota. Otherwise — once placed[auteur]
        # hits its target, the bucket gets skipped, and the gap-boost
        # never gets to peek a director-match candidate. Re-opening
        # auteur lets the next director film in its V3-rank queue
        # compete with its score multiplied by gap_boost_multiplier.
        auteur_reopen = gap_boost_multiplier > 1.0
        for b, target_count in target.items():
            if b != BUCKET_BEST_OVERALL and placed[b] >= target_count:
                if not (b == BUCKET_AUTEUR and auteur_reopen):
                    continue
            peek = _peek_next_eligible_for_bucket(
                queue_by_bucket[b],
                queue_pos[b],
                used_movie_ids,
                slot_index=slot_index,
                anchor_format_bucket=anchor_format_bucket,
                enforce_format_top_lock=enforce_format_top_lock,
                candidate_format_bucket_by_movie=candidate_format_bucket_by_movie,
                enforce_shorts_cap=enforce_shorts_cap,
                shorts_count=shorts_count,
                enforce_franchise_fatigue=enforce_franchise_fatigue,
                is_franchise_by_movie=is_franchise_by_movie,
                franchise_count=franchise_count,
                non_franchise_count=non_franchise_count,
            )
            if peek is None:
                continue
            c, pos_at = peek
            # V3.4.9 gap-boost: director-match candidates (any candidate
            # sharing a director with the anchor — no shape gate) get
            # their full score multiplied by ``gap_boost_multiplier`` for
            # this slot's MMR comparison. Non-director candidates use
            # their raw score. The actual returned ``score`` on the
            # candidate is left untouched; only the slot-local
            # ranking signal changes.
            effective_score = (
                c.score * gap_boost_multiplier if c.director_match else c.score
            )
            relevance = effective_score / best_score
            # Clamp deficit at 0: once a bucket has exceeded its target
            # via multi-bucket credit, its starvation pull falls to 0
            # and only relevance carries the adjustment.
            deficit_ratio = (
                max(0.0, (target_count - placed[b]) / target_count)
                if target_count > 0 else 0.0
            )
            adj = (1.0 - WEAVER_LAMBDA) * relevance + WEAVER_LAMBDA * deficit_ratio
            eligible.append((b, c, adj, pos_at, effective_score))

        if not eligible:
            break

        # Pick max adjusted score. Tie-break by effective score so the
        # gap-boost affects ties too — otherwise a boosted director
        # candidate would lose ties to a non-director candidate with
        # equal `adj` but higher raw `score`.
        eligible.sort(key=lambda x: (-x[2], -x[4], x[1].movie_id))
        b_drawn, c_picked, _, pos_at, _ = eligible[0]
        woven.append(c_picked)
        used_movie_ids.add(c_picked.movie_id)
        if candidate_format_bucket_by_movie.get(c_picked.movie_id) == "short":
            shorts_count += 1
        # V3.4.6: update fatigue counts after every placement so the
        # next slot's peek sees current ratio.
        if is_franchise_by_movie.get(c_picked.movie_id, False):
            franchise_count += 1
        else:
            non_franchise_count += 1
        # V3.4.9: update gap counter. Director-match placement resets;
        # any other placement increments.
        if c_picked.director_match:
            consecutive_non_director = 0
        else:
            consecutive_non_director += 1

        # Advance the drawn bucket's queue cursor past the picked candidate.
        queue_pos[b_drawn] = pos_at + 1

        # Multi-bucket full credit: bump placed[] for every bucket the
        # candidate is a member of (per V3.4 Decision 8). This prevents
        # the "best-overall surfaces 3 Nolan films, then auteur double-
        # downs with 3 more" failure mode by giving auteur its quota
        # credit for the films that already went via best-overall.
        memberships = bucket_memberships_by_movie.get(c_picked.movie_id, set()) | {
            BUCKET_BEST_OVERALL
        }
        for b_member in memberships:
            if b_member in target:
                placed[b_member] += 1

    # Past the top section: append remaining candidates in V3-rank order
    # so callers requesting limit > 10 still get a full list. The
    # franchise fatigue gate continues here using the SAME counters
    # accumulated during the top-section loop — the rule operates over
    # the whole result list, not per-section. Counters update as we
    # append so the gate naturally lifts as more non-franchise picks
    # come in, then re-engages if a streak of franchise candidates
    # would push the ratio back over threshold.
    if limit > top_section_cap:
        for c in base_sorted:
            if len(woven) >= limit:
                break
            if c.movie_id in used_movie_ids:
                continue
            if (
                enforce_franchise_fatigue
                and is_franchise_by_movie.get(c.movie_id, False)
                and franchise_count
                > FRANCHISE_FATIGUE_THRESHOLD * non_franchise_count
            ):
                continue
            woven.append(c)
            used_movie_ids.add(c.movie_id)
            if is_franchise_by_movie.get(c.movie_id, False):
                franchise_count += 1
            else:
                non_franchise_count += 1

    return woven[:limit], target


def _record_weave_targets(target_by_bucket: dict[str, int]) -> None:
    """Set `branch_weave_targets` on the current span as a JSON `{bucket: slots}` map.

    Records the DESIRED allocation `_compute_bucket_targets` set before weaving, NOT
    the realized per-slot draw. A signal bucket absent from the map never
    instantiated (its signal was below the gate); a bucket present with N slots was
    the reservation the weave aimed for. Note this can diverge from what actually
    landed: multi-bucket full credit lets a signal bucket's films enter via
    best_overall, so an instantiated bucket can draw 0 real seats while still showing
    its target here — that's intended, the target is the more useful reader signal.
    Iterated in `ALL_BUCKETS` order for stable output; best_overall (always present)
    leads. A no-op outside a traced request (the current span is then non-recording).
    """
    span = trace.get_current_span()
    ordered = {
        bucket: target_by_bucket[bucket]
        for bucket in ALL_BUCKETS
        if bucket in target_by_bucket
    }
    span.set_attribute(SIMILARITY_WEAVE_TARGETS, json.dumps(ordered))


async def _resolve_similarity_reference(reference: SimilarityReference) -> dict | None:
    """Resolve a single SimilarityReference to the full signal row of its
    most-popular matching movie, optionally filtered by an explicitly-stated
    release_year.

    Returns the winning movie's ``fetch_similarity_signal_rows`` row — the exact
    shape ``run_similar_movies_for_ids`` needs for an anchor — so the caller hands
    it straight through instead of discarding it and re-fetching the same row by ID.
    Returns None when nothing matches; callers in the multi-reference path drop
    those entries silently per the Step 0 contract.
    """
    title = reference.similar_search_title.strip()
    if not title:
        # SimilarityReference's pydantic constraint already enforces
        # min_length=1 on the title; this guard is defensive in case a
        # caller constructs the dataclass outside the schema.
        raise ValueError("similar_search_title must be non-empty.")

    pattern = escape_like(normalize_string(title))
    if not pattern:
        return None

    title_ids = await fetch_movie_ids_with_title_like(pattern)
    if not title_ids:
        return None

    rows = await fetch_similarity_signal_rows(list(title_ids))
    candidates = list(rows.values())
    if reference.release_year is not None:
        candidates = [
            row for row in candidates if _release_year(row) == reference.release_year
        ]
    if not candidates:
        return None

    candidates.sort(
        key=lambda row: (
            -(row.get("popularity_percentile") or -1.0),
            -(row.get("reception_score") or -1.0),
            -(_release_year(row) or -1),
            int(row["movie_id"]),
        )
    )
    return candidates[0]


async def _resolve_similarity_anchors(
    references: list[SimilarityReference],
) -> tuple[list[int], dict[int, dict], list[bool]]:
    """Resolve every reference in input order, dropping unresolvable
    ones, and de-dup the remainder while preserving first-seen order.

    A reference is "unresolvable" when the title doesn't match any
    movie in the catalog (or, if a release_year was supplied, no
    title-match has that year). Per the Step 0 contract those entries
    are silently dropped; only when EVERY reference fails does the
    similarity search end up empty.

    Per-reference lookups are independent (each one round-trips Postgres
    twice), so they run concurrently via ``asyncio.gather``. ``gather``
    preserves argument order in its return list, which is what the
    downstream dedupe relies on to keep first-seen ordering stable.

    Returns ``(anchor_ids, anchor_rows, per_ref_resolved)``: the deduped anchor
    list; an ``{anchor_id: signal_row}`` map carrying the winning rows already
    fetched during resolution, so the caller hands them to the engine instead of
    triggering a redundant re-fetch by ID; and a per-reference bool (index-aligned
    with ``references``) marking which reference titles resolved to an anchor — the
    requested-vs-resolved signal that catches a silently-dropped reference (a
    plausible-looking result set built from fewer anchors than the user named).
    """
    resolved_rows = await asyncio.gather(
        *(_resolve_similarity_reference(ref) for ref in references)
    )
    out: list[int] = []
    anchor_rows: dict[int, dict] = {}
    for row in resolved_rows:
        if row is None:
            continue
        anchor_id = int(row["movie_id"])
        # dict membership doubles as the dedupe set; first-seen order is kept by
        # the parallel `out` list (a dict preserves insertion order, but `out`
        # makes the ordering contract explicit for readers).
        if anchor_id in anchor_rows:
            continue
        anchor_rows[anchor_id] = row
        out.append(anchor_id)
    per_ref_resolved = [row is not None for row in resolved_rows]
    return out, anchor_rows, per_ref_resolved


async def run_similarity_search(
    flow_data: SimilarityFlowData,
    *,
    limit: int = 50,
    qdrant_limit: int = DEFAULT_QDRANT_LIMIT,
    metadata_filters: MetadataFilters | None = None,
) -> SimilarMoviesSearchResult:
    """Run Step-0's title-based similarity flow.

    Resolves every reference in flow_data.references to a tmdb_id and
    hands the deduped anchor list to run_similar_movies_for_ids, which
    routes single-anchor vs. multi-anchor pipelines internally. Returns
    an empty result when no reference resolves.

    ``metadata_filters`` (when active) constrains every candidate-
    generation lane and the Qdrant shape search to movies that pass the
    user's hard filters — matching the behavior of the other Step-0
    entity flows (exact_title, franchise, studio, person).
    """
    if not flow_data.should_be_searched:
        raise ValueError(
            "run_similarity_search called with should_be_searched=False; "
            "the caller must gate on Step 0's flow decision."
        )
    # Non-empty references when should_be_searched=True is enforced by
    # the Step 0 schema validator (validate_flow_titles_when_searched),
    # so we trust the contract here.
    anchor_ids, anchor_rows, per_ref_resolved = await _resolve_similarity_anchors(
        flow_data.references
    )
    _record_similarity_entities(flow_data.references, per_ref_resolved)
    if not anchor_ids:
        return SimilarMoviesSearchResult(
            anchor_movie_ids=[],
            ranked=[],
            active_anchor_types=[],
            debug=SimilarMoviesDebug(vector_space_weights={}),
        )
    # Hand the winning rows resolved above straight to the engine; anchor facets
    # were already fetched while disambiguating each title, so the engine's anchor
    # load fetches nothing on this path (see `run_similar_movies_for_ids`). The
    # engine records the flow-neutral `similarity.*` signal attributes itself, so
    # both this branch path and the pure /similarity_search endpoint get them with
    # no per-caller wiring — only the reference-resolution skeleton above is
    # query_search-specific.
    return await run_similar_movies_for_ids(
        anchor_ids,
        prefetched_anchor_rows=anchor_rows,
        limit=limit,
        qdrant_limit=qdrant_limit,
        metadata_filters=metadata_filters,
    )


def _record_similarity_entities(
    references: list[SimilarityReference],
    per_ref_resolved: list[bool],
) -> None:
    """Set the entity skeleton on the branch span: the reference titles, a
    per-reference resolved count (1 = the title resolved to an anchor, 0 = a
    silently-dropped reference), and the unresolved total."""
    span = trace.get_current_span()
    span.set_attribute(
        QUERY_SEARCH_BRANCH_ENTITIES,
        [ref.similar_search_title for ref in references],
    )
    span.set_attribute(
        QUERY_SEARCH_BRANCH_ENTITY_RESOLVED_COUNTS,
        [1 if resolved else 0 for resolved in per_ref_resolved],
    )
    span.set_attribute(
        QUERY_SEARCH_BRANCH_UNRESOLVED_ENTITY_COUNT,
        sum(1 for resolved in per_ref_resolved if not resolved),
    )


# Shape token for an anchor (or cohort fraction) that classifies into no
# reach×quality shape — the common "normal film" middle cell (reception 50–80,
# sub-100K reach, no award shift). Emitted as the explicit string `"none"`
# (single-anchor scalar) or a `"none"` key (multi-anchor cohesion map) rather
# than being omitted, so the shapeless verdict stays a visible, first-class value.
_ANCHOR_SHAPE_NONE = "none"

# Float-noise guard for the multi-anchor shapeless fraction (1 - sum of shaped
# fractions). Fractions are exact count/N ratios, so real "none" mass is well
# above this; the epsilon only suppresses a spurious ~0 key from FP rounding.
_SHAPE_NONE_EPSILON = 1e-9

# Multiplier/boost paths surfaced on `branch_additional_boosts` — signals that
# amplify scoring but aren't recoverable from the additive weights or the fetch
# map. Currently just the auteur multiplier, flagged by `director_signature`
# appearing in `active_anchor_types`. Extend the tuple as new boosts are added.
_ADDITIONAL_BOOST_TYPES: tuple[str, ...] = ("director_signature",)


def _set_json_map(span: object, name: str, mapping: dict) -> None:
    """Set a `{label: number}` map on the span as a single JSON-string attribute.

    OTel drops a raw dict attribute; a JSON string is kept and renders readably in
    Tempo/Grafana with label and value side by side. Keys are sorted for stable,
    diff-friendly output. Empty maps still emit `"{}"` (a visible non-empty string).
    """
    span.set_attribute(name, json.dumps({k: mapping[k] for k in sorted(mapping)}))


def _record_similarity_signals(result: SimilarMoviesSearchResult) -> None:
    """Set the flow-neutral `similarity.*` signal attributes on the CURRENT span,
    organized around the four reader questions: (1) which traits mattered, (2) which
    avenues fetched candidates and how many each returned, (3) how strong the scoring
    weights were, (4) which paths were active in the final weave. Called from the
    engine, so the current span is the `query_search.branch` span on the /query_search
    path and the FastAPI server span on the pure /similarity_search path. Map-shaped
    signals are single JSON-string attributes (see `_set_json_map`). The (1) block is
    flow-specific — single-anchor emits its additive weight modifiers + scalar shape,
    multi-anchor emits per-shape/lane/vector-space cohort cohesion — so the split
    mirrors how the two flows actually differ. Weave-seat counts are set deeper, in
    `_build_results`."""
    span = trace.get_current_span()
    debug = result.debug
    is_single = len(result.anchor_movie_ids) == 1

    # (2) Candidate-fetch avenues (fired lanes → result count) + deduped union.
    _set_json_map(span, SIMILARITY_RETRIEVAL_LANES, debug.retrieval_counts_by_lane)
    span.set_attribute(SIMILARITY_RETRIEVAL_TOTAL, debug.retrieval_total)

    # (3) Scoring weight strengths: additive lane weights + the 8-space shape mix.
    _set_json_map(span, SIMILARITY_LANE_WEIGHTS, debug.normalized_lane_weights)
    _set_json_map(
        span, SIMILARITY_VECTOR_SPACE_WEIGHTS, debug.vector_space_weights
    )

    # (4) Final-weave paths: seat map is set in `_build_results`; fallback flag here.
    span.set_attribute(
        SIMILARITY_LOW_COHESION_FALLBACK,
        bool(debug.low_cohesion_fallback_used),
    )

    # Multiplier boosts invisible in the weights/fetch map (currently the auteur
    # multiplier). Omit the attribute entirely when nothing fired.
    boosts = [b for b in _ADDITIONAL_BOOST_TYPES if b in result.active_anchor_types]
    if boosts:
        span.set_attribute(SIMILARITY_ADDITIONAL_BOOSTS, json.dumps(boosts))

    # (1) Traits marked important — flow-specific presentation.
    if is_single:
        # Additive lane-weight-delta modifiers enacted (cult_garbage / prestige /
        # franchise_dominant / source_material); derived from the delta table so it
        # stays in sync. Always set — `"[]"` explicitly signals "no modifiers".
        modifiers = [
            t for t in result.active_anchor_types if SINGLE_ANCHOR_ADJUSTMENTS.get(t)
        ]
        span.set_attribute(SIMILARITY_SHAPE_MODIFIERS, json.dumps(modifiers))
        # Scalar reach×quality shape; single-anchor cohesion is {shape: 1.0} or {}.
        shape_keys = list(debug.anchor_shape_cohesion)
        span.set_attribute(
            SIMILARITY_ANCHOR_SHAPE,
            shape_keys[0] if shape_keys else _ANCHOR_SHAPE_NONE,
        )
    else:
        # Cohort shape composition, dominant fraction first, with an explicit
        # `"none"` entry for the shapeless fraction so the map sums to 1.
        cohesion = dict(debug.anchor_shape_cohesion)
        none_fraction = 1.0 - sum(cohesion.values())
        # Keep full float precision (not rounded) so `none` matches the shaped
        # fractions' precision and the map sums to 1 within float epsilon.
        if none_fraction > _SHAPE_NONE_EPSILON:
            cohesion[_ANCHOR_SHAPE_NONE] = none_fraction
        ordered = dict(sorted(cohesion.items(), key=lambda kv: (-kv[1], kv[0])))
        span.set_attribute(
            SIMILARITY_ANCHOR_SHAPE_COHESION, json.dumps(ordered)
        )
        _set_json_map(span, SIMILARITY_LANE_COHESION, debug.lane_cohesion)
        _set_json_map(
            span, SIMILARITY_VECTOR_SPACE_COHESION, debug.vector_space_cohesion
        )


async def run_similar_movies_for_ids(
    tmdb_ids: list[int],
    *,
    prefetched_anchor_rows: dict[int, dict] | None = None,
    limit: int = 50,
    qdrant_limit: int = DEFAULT_QDRANT_LIMIT,
    quality_limit: int = DEFAULT_QUALITY_LIMIT,
    metadata_filters: MetadataFilters | None = None,
) -> SimilarMoviesSearchResult:
    """Run similar-movies search for one or more anchor TMDB IDs.

    This is the shared engine entry point reached two ways: the title flow
    (``run_similarity_search``) resolves each reference to its signal row and
    passes those rows via ``prefetched_anchor_rows``; the raw-ID callers (batch
    runner, direct API endpoint) pass only IDs. ``prefetched_anchor_rows`` lets a
    caller that already holds anchor signal rows hand them in so the anchor facet
    load queries only the IDs it wasn't given — on the title path, none — while
    raw-ID callers fetch every anchor here exactly as before.

    ``metadata_filters`` is threaded into every candidate-generation lane
    (Postgres lanes via inline / direct ``movie_card`` filter clauses; the
    Qdrant shape search via ``build_qdrant_filter``) so a filter that's
    inactive at the request boundary stays a no-op end to end.
    """
    anchor_ids = list(dict.fromkeys(int(mid) for mid in tmdb_ids))
    if not anchor_ids:
        raise ValueError("tmdb_ids must contain at least one movie ID.")

    # Reuse any anchor rows the caller already fetched (title path), and query only
    # the anchors we weren't handed. `fetch_similarity_signal_rows([])` short-circuits
    # to `{}` with no round-trip, so the title path fetches — and traces — nothing here.
    anchor_id_set = set(anchor_ids)
    supplied_rows = {
        mid: row
        for mid, row in (prefetched_anchor_rows or {}).items()
        if mid in anchor_id_set
    }
    missing_ids = [mid for mid in anchor_ids if mid not in supplied_rows]

    # The remaining prefetches are independent — collapsing their RTTs into
    # max(RTT). The anchor-row fetch (empty on the title path) joins the same
    # gather so raw-ID callers still overlap all four reads.
    (
        fetched_rows,
        vectors_by_anchor,
        studio_entries_by_company_id,
        director_terms_by_anchor,
    ) = await asyncio.gather(
        fetch_similarity_signal_rows(missing_ids),
        _load_anchor_vectors(anchor_ids),
        _load_studio_entries_by_company_id(),
        fetch_director_term_ids_for_movies(anchor_ids),
    )
    anchor_rows = {**supplied_rows, **fetched_rows}
    # Fatal if any anchor still lacks a row — a supplied-but-stale ID or a raw ID
    # with no movie_card row both surface here.
    missing = [mid for mid in anchor_ids if mid not in anchor_rows]
    if missing:
        raise LookupError(f"movie_card rows not found for tmdb_ids={missing}")

    # anchor_count is the single-vs-multi discriminator that selects the pipeline
    # below and governs how every downstream similarity signal reads. Set it on the
    # current span (the query_search.branch span on the branch path; the FastAPI
    # server span on the /similarity_search endpoint) so both callers carry it;
    # it stands in for the query_search entity skeleton, which the pure endpoint
    # can't produce (anchors are supplied, not resolved). No-op offline.
    span = trace.get_current_span()
    span.set_attribute(SIMILARITY_ANCHOR_COUNT, len(anchor_ids))

    if len(anchor_ids) == 1:
        result = await _run_single_anchor_similarity(
            anchor_ids[0],
            anchor_rows,
            vectors_by_anchor,
            studio_entries_by_company_id,
            director_terms_by_anchor,
            limit=limit,
            qdrant_limit=qdrant_limit,
            quality_limit=quality_limit,
            metadata_filters=metadata_filters,
        )
    else:
        result = await _run_multi_anchor_similarity(
            anchor_ids,
            anchor_rows,
            vectors_by_anchor,
            studio_entries_by_company_id,
            director_terms_by_anchor,
            limit=limit,
            qdrant_limit=qdrant_limit,
            quality_limit=quality_limit,
            metadata_filters=metadata_filters,
        )

    # Record the flow-neutral `similarity.*` signal attributes here in the engine so
    # BOTH callers get them: the /query_search similarity branch (recording onto its
    # branch span) and the pure /similarity_search endpoint (recording onto the
    # server span). Relocated from `run_similarity_search`, which previously called
    # this only on the branch path.
    _record_similarity_signals(result)
    return result


# Per-lane candidate-fetch instrumentation. Each Postgres retrieval lane in the
# similarity flow runs concurrently inside `asyncio.gather`; `gather` wraps every
# coroutine in its own Task, and a Task copies the current OTel context at creation
# time (the branch span is current then). So a span started inside a wrapped fetch
# nests under the branch span, and the fetch's own auto-instrumented SQL span nests
# under THAT — no cross-lane context bleed. The qdrant shape probe is deliberately
# excluded (its params never vary; it carries its own `similarity_qdrant` span).
_FetchResultT = TypeVar("_FetchResultT", bound=Sized)


async def _traced_lane_fetch(
    lane: str,
    match: dict[str, object],
    coro: Awaitable[_FetchResultT],
) -> _FetchResultT:
    """Wrap one similarity candidate-fetch coroutine in a `similarity_fetch` span.

    Records the target `lane`, the concrete `match` values the lane queried on (the
    bound IN-list IDs the auto-instrumented SQL span parameterizes away, as a JSON
    object since the keys vary by lane), and the returned `result_count`, then awaits
    and returns the fetch result unchanged. Behavior-preserving: the result and any
    raised exception propagate exactly as the bare `await coro` would.
    """
    with tracer.start_as_current_span(SIMILARITY_FETCH) as span:
        span.set_attribute(SIMILARITY_FETCH_LANE, lane)
        span.set_attribute(
            SIMILARITY_FETCH_MATCH, json.dumps(match, sort_keys=True)
        )
        result = await coro
        span.set_attribute(SIMILARITY_FETCH_RESULT_COUNT, len(result))
        return result


def _franchise_match(
    lineage_entry_ids: set[int],
    shared_universe_entry_ids: set[int],
    subgroup_entry_ids: set[int],
) -> dict[str, object]:
    """Build the franchise-lane `match` object, omitting empty ID dimensions so the
    span shows only the franchise axes the query actually filtered on."""
    match: dict[str, object] = {}
    if lineage_entry_ids:
        match["lineage_entry_ids"] = sorted(lineage_entry_ids)
    if shared_universe_entry_ids:
        match["shared_universe_entry_ids"] = sorted(shared_universe_entry_ids)
    if subgroup_entry_ids:
        match["subgroup_entry_ids"] = sorted(subgroup_entry_ids)
    return match


def _themes_recall_match(traits: set[tuple[int, int]]) -> dict[str, object]:
    """Build the themes-recall `match` object, splitting the (kind, trait_id) pairs
    into the same per-kind ID legs the recall SQL fires on; empty legs omitted."""
    keyword_ids = sorted(t for (k, t) in traits if k == TRAIT_KIND_OVERALL_KEYWORD)
    concept_tag_ids = sorted(t for (k, t) in traits if k == TRAIT_KIND_CONCEPT_TAG)
    genre_ids = sorted(t for (k, t) in traits if k == TRAIT_KIND_TMDB_GENRE)
    match: dict[str, object] = {}
    if keyword_ids:
        match["keyword_ids"] = keyword_ids
    if concept_tag_ids:
        match["concept_tag_ids"] = concept_tag_ids
    if genre_ids:
        match["genre_ids"] = genre_ids
    return match


async def _run_single_anchor_similarity(
    anchor_id: int,
    anchor_rows: dict[int, dict],
    vectors_by_anchor: dict[int, dict[VectorName, list[float]]],
    studio_entries_by_company_id: dict[int, list[StudioSimilarityEntry]],
    director_terms_by_anchor: dict[int, set[int]],
    *,
    limit: int,
    qdrant_limit: int,
    quality_limit: int,
    metadata_filters: MetadataFilters | None = None,
) -> SimilarMoviesSearchResult:
    anchor_row = anchor_rows[anchor_id]
    quality_bucket = _quality_bucket(anchor_row)
    cult_or_prestige = quality_bucket in {"cult_garbage", "prestige"}

    anchor_directors = director_terms_by_anchor.get(anchor_id, set())
    anchor_lineage, anchor_universe, anchor_subgroups = _franchise_traits(anchor_row)
    anchor_source_ids = _as_int_set(anchor_row.get("source_material_type_ids"))
    anchor_studio_entries = _active_studio_entries(
        anchor_row, studio_entries_by_company_id
    )
    anchor_studio_company_ids = {entry.company_id for entry in anchor_studio_entries}
    anchor_medium_tags = _medium_tags_for_movie(anchor_row)
    anchor_format_bucket = format_bucket(anchor_row.get("keyword_ids") or ())
    anchor_themes_traits = _themes_traits_for_movie(anchor_row)

    # Pre-fetches for V3 lane data. The V2 director_strengths and
    # franchise_confidence MV reads are retired (V3 §2.1, §2.2); the
    # auteur set replaces director_strength as the lane gate, and the
    # franchise structural matrix replaces the consistency check.
    # Source IDFs use the unified trait-IDF MV filtered to kind=4.
    # Themes IDFs (V3 §2.3 single-anchor + §2.6 rare-keyword) read the
    # unified MV at kinds 0/2/3 (overall_keyword / concept_tag /
    # tmdb_genre). Medium IDFs are loaded lazily once per process and
    # cached. None of these depend on the candidate pool, so they run
    # in parallel with shape search and the candidate-generation lanes.
    auteur_term_ids_task = fetch_auteur_term_ids()
    source_idf_task = fetch_trait_idfs(
        [(TRAIT_KIND_SOURCE_MATERIAL, t) for t in anchor_source_ids]
    )
    themes_idf_task = fetch_trait_idfs(list(anchor_themes_traits))
    medium_idf_task = load_medium_idfs()

    shape_task = _run_single_anchor_shape_search(
        anchor_id,
        vectors_by_anchor.get(anchor_id, {}),
        qdrant_limit=qdrant_limit,
        metadata_filters=metadata_filters,
    )
    director_task = fetch_director_movie_terms(
        anchor_directors, metadata_filters=metadata_filters,
    )
    franchise_task = fetch_similarity_franchise_candidates(
        lineage_entry_ids=anchor_lineage,
        shared_universe_entry_ids=anchor_universe,
        subgroup_entry_ids=anchor_subgroups,
        metadata_filters=metadata_filters,
    )
    studio_task = fetch_movie_ids_by_production_company_ids(
        anchor_studio_company_ids, metadata_filters=metadata_filters,
    )
    source_task = fetch_similarity_source_candidates(
        anchor_source_ids, metadata_filters=metadata_filters,
    )
    # Quality candidate lane is recall-repair only — middle-bucket flow
    # leans on candidates surfaced by other lanes plus shape. When hard
    # filters are active the LIMIT clause caps the result before the
    # filter applies, so the post-filter set can be much thinner than
    # `quality_limit`. Over-fetch 2× when filters are active so the
    # eligible-post-filter pool stays usefully sized.
    effective_quality_limit = (
        quality_limit * 2 if metadata_filters is not None else quality_limit
    )
    quality_task = (
        fetch_similarity_quality_candidates(
            bucket=quality_bucket,
            limit=effective_quality_limit,
            metadata_filters=metadata_filters,
        )
        if cult_or_prestige
        else _empty_set()
    )

    # V3.1 themes recall: fetch candidates whose shared anchor traits
    # qualify by single-trait rarity OR combined-IDF sum. The split by
    # kind matches the `_themes_traits_for_movie` builder; per-kind
    # legs let the SQL skip families the anchor doesn't carry.
    themes_recall_task = fetch_movie_ids_by_themes_recall(
        keyword_ids=[
            tid for (kind, tid) in anchor_themes_traits
            if kind == TRAIT_KIND_OVERALL_KEYWORD
        ],
        concept_tag_ids=[
            tid for (kind, tid) in anchor_themes_traits
            if kind == TRAIT_KIND_CONCEPT_TAG
        ],
        genre_ids=[
            tid for (kind, tid) in anchor_themes_traits
            if kind == TRAIT_KIND_TMDB_GENRE
        ],
        metadata_filters=metadata_filters,
    )

    # Wrap each Postgres candidate lane that actually runs in its own span so the
    # trace shows the target lane and the concrete IDs it matched on (the qdrant
    # shape probe is excluded — its params never vary). Gate on the same seed
    # predicates as `retrieval_counts` below, so a span appears iff that lane's
    # query fired.
    if anchor_directors:
        director_task = _traced_lane_fetch(
            "director", {"director_term_ids": sorted(anchor_directors)}, director_task
        )
    if anchor_lineage or anchor_universe or anchor_subgroups:
        franchise_task = _traced_lane_fetch(
            "franchise",
            _franchise_match(anchor_lineage, anchor_universe, anchor_subgroups),
            franchise_task,
        )
    if anchor_studio_company_ids:
        studio_task = _traced_lane_fetch(
            "studio", {"company_ids": sorted(anchor_studio_company_ids)}, studio_task
        )
    if anchor_source_ids:
        source_task = _traced_lane_fetch(
            "source",
            {"source_material_type_ids": sorted(anchor_source_ids)},
            source_task,
        )
    if cult_or_prestige:
        quality_task = _traced_lane_fetch(
            "quality",
            {"bucket": quality_bucket, "limit": effective_quality_limit},
            quality_task,
        )
    if anchor_themes_traits:
        themes_recall_task = _traced_lane_fetch(
            "themes_recall",
            _themes_recall_match(anchor_themes_traits),
            themes_recall_task,
        )

    (
        (shape_scores, vector_space_weights),
        director_candidate_terms,
        franchise_candidate_ids,
        studio_candidate_ids,
        source_candidate_ids,
        quality_candidate_ids,
        auteur_term_ids,
        source_idfs_pairs,
        themes_idfs,
        medium_idfs,
        themes_recall_candidate_ids,
    ) = await asyncio.gather(
        shape_task,
        director_task,
        franchise_task,
        studio_task,
        source_task,
        quality_task,
        auteur_term_ids_task,
        source_idf_task,
        themes_idf_task,
        medium_idf_task,
        themes_recall_task,
    )

    # Source IDFs come back keyed by (kind, trait_id); flatten to trait_id
    # only for the lane scorer (kind is constant within this lane).
    source_idfs: dict[int, float] = {
        trait_id: idf for (_, trait_id), idf in source_idfs_pairs.items()
    }

    # V2 selective rare-medium retrieval: pull in additional candidates
    # sharing any rare medium tag with the anchor (idf >= threshold) so
    # e.g. a stop-motion anchor still surfaces other stop-motion films
    # even when the centroid-driven shape lane misses them. LIVE_ACTION
    # is excluded — it covers the bulk of the catalog so an overlap
    # query would explode the pool with no signal.
    rare_medium_tags = {
        tag
        for tag in anchor_medium_tags
        if medium_idfs.get(tag, 0.0) >= RARE_MEDIUM_IDF_THRESHOLD
        and tag != LIVE_ACTION_TAG_ID
    }
    rare_medium_candidate_ids: set[int] = (
        await _traced_lane_fetch(
            "rare_medium",
            {"medium_tag_ids": sorted(rare_medium_tags)},
            fetch_movie_ids_by_overall_keywords(
                list(rare_medium_tags), metadata_filters=metadata_filters,
            ),
        )
        if rare_medium_tags
        else set()
    )

    # V3 §2.1 director_signature anchor type — fires when at least one
    # of the anchor's directors is on the curated auteur list.
    has_director_signature = bool(anchor_directors & auteur_term_ids)

    # V3 §2.2 franchise_dominant: simplified to "anchor has at least
    # one lineage_entry_id" (i.e., is part of any franchise). The V2
    # confidence/consistency gating is retired — the structural matrix
    # already scales scores by overlap quality so direct-to-DVD
    # spinoffs can't ride the +0.18 weight bump into the top.
    is_franchise_anchor = bool(anchor_lineage)

    active_anchor_types: list[AnchorType] = ["standard_shape"]
    if quality_bucket in {"cult_garbage", "prestige"}:
        active_anchor_types.append(quality_bucket)  # type: ignore[arg-type]
    if is_franchise_anchor:
        active_anchor_types.append("franchise_dominant")
    if anchor_studio_entries:
        active_anchor_types.append("studio_lineage")
    if anchor_source_ids:
        active_anchor_types.append("source_material")
    if has_director_signature:
        active_anchor_types.append("director_signature")

    raw_lane_weights, lane_weights = _single_anchor_lane_weights(
        active_anchor_types,
        quality_bucket=quality_bucket,
    )

    candidate_ids = set(shape_scores)
    candidate_ids.update(director_candidate_terms)
    candidate_ids.update(franchise_candidate_ids)
    candidate_ids.update(studio_candidate_ids)
    candidate_ids.update(source_candidate_ids)
    candidate_ids.update(quality_candidate_ids)
    candidate_ids.update(rare_medium_candidate_ids)
    candidate_ids.update(themes_recall_candidate_ids)  # V3.1 themes recall path
    candidate_ids.discard(anchor_id)

    # Retrieval-side pool sizes for the branch span. Key a lane only when its
    # seed was non-empty (the query actually ran) — a fired-but-empty lane stays
    # present at 0 (an informative retrieval gap), a lane with no seed is absent.
    # shape always runs. Mirrors the per-lane fetch conditions above.
    retrieval_counts: dict[str, int] = {"shape": len(shape_scores)}
    if anchor_directors:
        retrieval_counts["director"] = len(director_candidate_terms)
    if anchor_lineage or anchor_universe or anchor_subgroups:
        retrieval_counts["franchise"] = len(franchise_candidate_ids)
    if anchor_studio_company_ids:
        retrieval_counts["studio"] = len(studio_candidate_ids)
    if anchor_source_ids:
        retrieval_counts["source"] = len(source_candidate_ids)
    if cult_or_prestige:
        retrieval_counts["quality"] = len(quality_candidate_ids)
    if anchor_themes_traits:
        retrieval_counts["themes_recall"] = len(themes_recall_candidate_ids)
    if rare_medium_tags:
        retrieval_counts["rare_medium"] = len(rare_medium_candidate_ids)
    retrieval_total = len(candidate_ids)

    # Candidate-side reads run in parallel — independent Postgres
    # queries with no dependency between them. V3.3.2: always fetch
    # award signals for both anchor and candidates — the shape
    # classifier needs them for picture-level prestige lift (80 → 65)
    # and bad-Razzie-WIN poor ceiling lift (50 → 60).
    candidate_rows, award_signals = await asyncio.gather(
        fetch_similarity_signal_rows(list(candidate_ids)),
        fetch_similarity_award_signals(list(candidate_ids) + [anchor_id]),
    )
    # One-pass parse of frequently-reused row derivations — see
    # _parse_candidate_rows. Downstream orchestrator-level lookups
    # (format bucket, medium tags, country tags, franchise pool) read
    # from this dict instead of re-parsing each row 2-3 times.
    parsed_by_movie = _parse_candidate_rows(candidate_rows)

    # ----- Per-lane scoring -----
    # Director: V3 single-anchor — 0.20 absolute contribution iff the
    # anchor's director is in the auteur set AND the candidate shares
    # that director. Non-auteur anchors leave the director lane silent
    # (V3 §2.1: "for non-auteurs, franchise/shape already cover").
    director_scores = _single_anchor_director_score(
        director_candidate_terms,
        anchor_directors,
        auteur_term_ids,
    )

    # Franchise: V3 §2.2 structural matrix — 5-tier rule on
    # (lineage match, subgroup overlap, universe overlap). The V2
    # multiplicative low-confidence path is retired; every hit now
    # flows through the additive lane.
    raw_franchise_scores = {
        movie_id: _franchise_score_v2(anchor_row, row)
        for movie_id, row in candidate_rows.items()
        if movie_id in franchise_candidate_ids
    }

    # Studio: scored on every studio-candidate; the multiplier path filters
    # by shape gate, so we don't need to do that here.
    studio_scores = {
        movie_id: _studio_score(
            anchor_row,
            row,
            anchor_studio_entries,
            _active_studio_entries(row, studio_entries_by_company_id),
        )
        for movie_id, row in candidate_rows.items()
        if movie_id in studio_candidate_ids
    }

    # Source: V2 IDF-weighted max over shared types.
    source_scores = {
        movie_id: _source_score_idf(
            anchor_source_ids,
            _as_int_set(row.get("source_material_type_ids")),
            source_idfs,
        )
        for movie_id, row in candidate_rows.items()
    }

    # Quality: V2 always-on per-bucket formula across every candidate row.
    quality_scores = {
        movie_id: _quality_score_v2(
            quality_bucket, row, award_signals.get(movie_id)
        )
        for movie_id, row in candidate_rows.items()
    }

    # Format: V2 binary same-bucket-or-not, every candidate. Uses the
    # precomputed format_bucket from parsed_by_movie so we don't
    # re-parse keyword_ids per candidate (dedupes with the format
    # bucket lookup feeding _build_results below).
    format_scores = {
        movie_id: 1.0 if parsed.format_bucket_value == anchor_format_bucket else 0.0
        for movie_id, parsed in parsed_by_movie.items()
    }

    # Medium multiplier per candidate (computed once, applied in build).
    medium_multiplier_by_movie: dict[int, float] = {}
    if anchor_medium_tags:
        for movie_id, parsed in parsed_by_movie.items():
            medium_multiplier_by_movie[movie_id] = _medium_multiplier(
                anchor_medium_tags, parsed.medium_tags
            )

    # V3 §2.2: every franchise hit goes through the additive lane —
    # the V2 high/low-confidence split is retired.
    franchise_additive: dict[int, float] = {
        movie_id: score
        for movie_id, score in raw_franchise_scores.items()
        if score > 0.0
    }

    # V3 §2.4: country/language multiplier extended to single-anchor.
    # The anchor's own country tags become the consensus set; a
    # candidate intersecting them is "match" (×1.05), a candidate with
    # any country tags but none in common is "mismatch" (×0.75).
    # Candidates with no country tags get neither boost nor penalty
    # (preserves the V2 "we can't tell" semantics). Inlined here using
    # parsed_by_movie so we don't re-parse keyword_ids → country_set.
    anchor_country_tags = frozenset(
        country_set(_as_int_set(anchor_row.get("keyword_ids")))
    )
    country_consensus_match_by_movie: dict[int, bool] = {}
    if anchor_country_tags:
        for mid, parsed in parsed_by_movie.items():
            if not parsed.country_tags:
                continue
            country_consensus_match_by_movie[mid] = bool(
                parsed.country_tags & anchor_country_tags
            )

    # V3 §2.3: themes lane extended to single-anchor — anchor's own
    # trait pool drives the denominator.
    single_themes_scores = _single_anchor_themes_scores(
        anchor_traits=anchor_themes_traits,
        candidate_rows=candidate_rows,
        idf_lookup=themes_idfs,
    )

    # V3.4.4: rare_keyword lane removed entirely. Tag-overlap is
    # scored via the themes lane (with V3.4.4 compounding bonus); the
    # rare_keyword bucket in the weaver pulls candidates that share
    # genuinely-rare (idf >= 0.55) traits with the anchor for
    # exploration, gated by `_compute_single_anchor_bucket_data`.

    # V3 §3.1 shorts harsh downrank requires per-candidate format
    # bucket so `_build_results` can multiply combined score by 0.30
    # for cross-format candidates and `_weave_candidates` can enforce
    # the max-1 hard cap. Pulled from the parsed slice — same value
    # `format_scores` already used, no second `format_bucket` call.
    candidate_format_bucket_by_movie: dict[int, FormatBucket] = {
        mid: parsed.format_bucket_value
        for mid, parsed in parsed_by_movie.items()
    }

    lane_scores: dict[LaneName, dict[int, float]] = {
        "shape": shape_scores,
        "director": {mid: s for mid, s in director_scores.items() if s > 0.0},
        "franchise": franchise_additive,
        # studio appears in lane_scores only for debug visibility — its
        # contribution to combined score is the multiplier path.
        "studio": {},
        "source": {mid: s for mid, s in source_scores.items() if s > 0.0},
        "quality": {mid: s for mid, s in quality_scores.items() if s > 0.0},
        "format": {mid: s for mid, s in format_scores.items() if s > 0.0},
        # V3 §2.3: themes lane extended to single-anchor.
        "themes": single_themes_scores,
        # Cast / specific_award stay multi-only.
        "cast": {},
        "specific_award": {},
    }

    # V3.1 single-anchor director floor: build a per-candidate ratio
    # dict where ratio = 1.0 iff the candidate has any auteur director
    # match (director_scores is keyed only on auteur matches per
    # _single_anchor_director_score). Threshold is also 1.0, so any
    # match qualifies; non-matches stay at 0.0.
    director_floor_max_ratio_single = {
        movie_id: 1.0 for movie_id in director_scores
    }

    # V3.3 shape multiplier — single-anchor: cohort cohesion is binary
    # (anchor either has a shape or doesn't). Classify the anchor and
    # every candidate using movie_card row data + award signals (V3.3.2).
    anchor_shape = _classify_shape(anchor_row, award_signals.get(anchor_id))
    anchor_shape_cohesion: dict[str, float] = (
        {anchor_shape: 1.0} if anchor_shape is not None else {}
    )
    candidate_shape_by_movie: dict[int, str | None] = {
        movie_id: _classify_shape(row, award_signals.get(movie_id))
        for movie_id, row in candidate_rows.items()
    }

    # V3.4 bucket-weaver: derive per-anchor signal scores + per-candidate
    # bucket memberships from the lane data already computed. Lead-actor
    # bucket is multi-only; we pass empty dicts for the multi-anchor-only
    # arguments. The weaver collapses to "best-overall takes all 10 slots
    # in V3-rank order" when no signal-bucket clears WEAVER_BUCKET_INSTANTIATE_MIN.
    bucket_signals, bucket_memberships = _compute_single_anchor_bucket_data(
        anchor_row=anchor_row,
        anchor_directors=anchor_directors,
        auteur_term_ids=auteur_term_ids,
        anchor_themes_traits=anchor_themes_traits,
        themes_idfs=themes_idfs,
        candidate_rows=candidate_rows,
        director_candidate_terms=director_candidate_terms,
        franchise_candidate_ids=franchise_candidate_ids,
    )

    # V3.4.6 franchise fatigue. Define "franchise" broadly: candidate
    # shares any lineage_entry_id OR shared_universe_entry_id with the
    # anchor (cross-matching across both kinds — anchor's lineage may
    # match candidate's universe and vice versa). Subgroup tags are
    # excluded by design; per the user spec, fatigue gates on the
    # broadest sense of "this candidate is part of the anchor's
    # franchise". The dict feeds the weaver's hard-ban check.
    anchor_franchise_pool: set[int] = anchor_lineage | anchor_universe
    is_franchise_by_movie: dict[int, bool] = {}
    if anchor_franchise_pool:
        for movie_id, parsed in parsed_by_movie.items():
            is_franchise_by_movie[movie_id] = bool(
                parsed.franchise_pool & anchor_franchise_pool
            )


    ranked, counts = _build_results(
        anchor_ids=[anchor_id],
        lane_scores=lane_scores,
        lane_weights=lane_weights,
        limit=limit,
        studio_score_by_movie=studio_scores,
        medium_multiplier_by_movie=medium_multiplier_by_movie,
        country_consensus_match_by_movie=country_consensus_match_by_movie,
        director_floor_max_ratio_by_movie=director_floor_max_ratio_single,
        director_floor_ratio_threshold=DIRECTOR_FLOOR_SINGLE_RATIO_THRESHOLD,
        director_floor_shape_gate=DIRECTOR_FLOOR_SINGLE_SHAPE_GATE,
        director_floor_magnitude=DIRECTOR_FLOOR_SINGLE_MAGNITUDE,
        candidate_format_bucket_by_movie=candidate_format_bucket_by_movie,
        anchor_active_format_bucket=anchor_format_bucket,
        anchor_format_bucket=anchor_format_bucket,
        enforce_format_top_lock=True,
        anchor_shape_cohesion=anchor_shape_cohesion,
        candidate_shape_by_movie=candidate_shape_by_movie,
        bucket_signals=bucket_signals,
        bucket_memberships_by_movie=bucket_memberships,
        enforce_franchise_fatigue=True,
        is_franchise_by_movie=is_franchise_by_movie,
        enforce_director_gap_boost=True,
    )
    return SimilarMoviesSearchResult(
        anchor_movie_ids=[anchor_id],
        ranked=ranked,
        active_anchor_types=active_anchor_types,
        debug=SimilarMoviesDebug(
            vector_space_weights=vector_space_weights,
            raw_lane_weights=raw_lane_weights,
            normalized_lane_weights=lane_weights,
            candidate_counts_by_lane=counts,
            retrieval_counts_by_lane=retrieval_counts,
            retrieval_total=retrieval_total,
            anchor_format_bucket=anchor_format_bucket,
            anchor_medium_tags=sorted(anchor_medium_tags),
            anchor_shape_cohesion=anchor_shape_cohesion,
        ),
    )


async def _run_multi_anchor_similarity(
    anchor_ids: list[int],
    anchor_rows: dict[int, dict],
    vectors_by_anchor: dict[int, dict[VectorName, list[float]]],
    studio_entries_by_company_id: dict[int, list[StudioSimilarityEntry]],
    director_terms_by_anchor: dict[int, set[int]],
    *,
    limit: int,
    qdrant_limit: int,
    quality_limit: int,
    metadata_filters: MetadataFilters | None = None,
) -> SimilarMoviesSearchResult:
    """V2 multi-anchor similarity flow.

    Adds five new lanes/multipliers vs. V1: themes (IDF over keyword +
    concept + genre), cast (top-3-billed overlap), specific_award (3-tier
    category-tag taxonomy), country/language coherence multiplier, and
    shape-lane scaling driven by mean vector cohesion. Source lane is
    upgraded to per-anchor IDF weighting. Studio is removed from the
    additive sum and applied as a multiplier (consistent with single-
    anchor V2).
    """
    n = len(anchor_ids)

    # Per-anchor trait sets — built once and reused for cohesion +
    # candidate scoring. Order matches anchor_ids so element i is anchor
    # i's traits.
    anchor_studio_entries = {
        anchor_id: _active_studio_entries(row, studio_entries_by_company_id)
        for anchor_id, row in anchor_rows.items()
    }
    director_trait_sets = [
        set(director_terms_by_anchor.get(anchor_id, set()))
        for anchor_id in anchor_ids
    ]
    franchise_trait_sets = [
        set().union(*_franchise_traits(anchor_rows[anchor_id]))
        for anchor_id in anchor_ids
    ]
    studio_trait_sets = [
        {entry.company_id for entry in anchor_studio_entries[anchor_id]}
        for anchor_id in anchor_ids
    ]
    source_trait_sets = [
        _as_int_set(anchor_rows[anchor_id].get("source_material_type_ids"))
        for anchor_id in anchor_ids
    ]
    quality_trait_sets: list[set[str]] = [
        {bucket}
        if (bucket := _quality_bucket(anchor_rows[anchor_id]))
        in {"cult_garbage", "prestige"}
        else set()
        for anchor_id in anchor_ids
    ]
    themes_trait_sets = [
        _themes_traits_for_movie(anchor_rows[anchor_id]) for anchor_id in anchor_ids
    ]
    format_trait_sets: list[set[str]] = [
        {format_bucket(anchor_rows[anchor_id].get("keyword_ids") or ())}
        for anchor_id in anchor_ids
    ]
    country_trait_sets = [
        country_set(anchor_rows[anchor_id].get("keyword_ids") or ())
        for anchor_id in anchor_ids
    ]

    # Repeated traits — cohesion uses these across every lane.
    # Sync cohesion for every lane that doesn't need a DB read. cast and
    # specific_award are added below once their anchor-side fetches return.
    cohesion_by_lane = {
        "director": _metadata_cohesion(director_trait_sets),
        "franchise": _metadata_cohesion(franchise_trait_sets),
        "studio": _metadata_cohesion(studio_trait_sets),
        "source": _metadata_cohesion(source_trait_sets),
        "quality": _metadata_cohesion(quality_trait_sets),
        "themes": _metadata_cohesion(themes_trait_sets),
        "format": _metadata_cohesion(format_trait_sets),
    }

    repeated_quality_bucket = _repeated_quality_bucket(quality_trait_sets)
    repeated_format_bucket = _repeated_format_bucket(format_trait_sets)
    consensus_countries = _consensus_country_set(country_trait_sets)

    # Multi-anchor candidate generation. director/franchise/studio/source/
    # quality candidate fetches gate on the sync cohesion above so we don't
    # waste DB calls on lanes with no repeated traits. cast and award are
    # cheap and small (~3 rows per anchor, ≤ N total awards), so they fire
    # unconditionally — the second-round candidate-side cast/award reads
    # later are still gated on the cohesion derived from these results.
    director_terms = set().union(*director_trait_sets) if director_trait_sets else set()
    franchise_traits = (
        set().union(*franchise_trait_sets) if franchise_trait_sets else set()
    )
    studio_company_ids = (
        set().union(*studio_trait_sets) if studio_trait_sets else set()
    )
    source_ids = set().union(*source_trait_sets) if source_trait_sets else set()

    shape_task = _run_multi_anchor_shape_search(
        anchor_ids,
        vectors_by_anchor,
        qdrant_limit=qdrant_limit,
        metadata_filters=metadata_filters,
    )
    # V3 §2.1 fires the multi-anchor director lane on EITHER director
    # cohesion (≥2 anchors share a director) OR auteur presence (any
    # anchor's director is curated, even with M_d=1). The latter is
    # what surfaces e.g. Tenet for a 3-anchor cohort where only
    # Inception is by a curated director — without this, the V3 unified
    # rule's M_d=1 + auteur path would silently miss because the
    # candidate-fetching task returned an empty dict.
    #
    # `fetch_auteur_term_ids` is module-cached after the first call, but
    # the first call per process otherwise blocks the parallel gather.
    # Fold the auteur fetch into a composite task: await auteur, decide
    # whether to fire director, then await director if needed. This keeps
    # the auteur read in parallel with the shape/franchise/studio tasks
    # while preserving the gate (no speculative director fetch in the
    # rare case neither cohesion nor auteur holds).
    director_cohesion_active = cohesion_by_lane["director"] > 0.0

    async def _fetch_auteur_and_director() -> tuple[set[int], bool, dict[int, set[int]]]:
        auteur = await fetch_auteur_term_ids()
        anchor_has_auteur = bool(director_terms & auteur)
        if director_cohesion_active or anchor_has_auteur:
            director_result = await _traced_lane_fetch(
                "director",
                {"director_term_ids": sorted(director_terms)},
                fetch_director_movie_terms(
                    director_terms, metadata_filters=metadata_filters,
                ),
            )
        else:
            director_result = {}
        return auteur, anchor_has_auteur, director_result

    director_combo_task = _fetch_auteur_and_director()
    franchise_task = (
        _traced_lane_fetch(
            "franchise",
            _franchise_match(franchise_traits, set(), set()),
            fetch_similarity_franchise_candidates(
                lineage_entry_ids=franchise_traits,
                shared_universe_entry_ids=set(),
                subgroup_entry_ids=set(),
                metadata_filters=metadata_filters,
            ),
        )
        if cohesion_by_lane["franchise"] > 0.0
        else _empty_set()
    )
    studio_task = (
        _traced_lane_fetch(
            "studio",
            {"company_ids": sorted(studio_company_ids)},
            fetch_movie_ids_by_production_company_ids(
                studio_company_ids, metadata_filters=metadata_filters,
            ),
        )
        if cohesion_by_lane["studio"] > 0.0
        else _empty_set()
    )
    source_task = (
        _traced_lane_fetch(
            "source",
            {"source_material_type_ids": sorted(source_ids)},
            fetch_similarity_source_candidates(
                source_ids, metadata_filters=metadata_filters,
            ),
        )
        if cohesion_by_lane["source"] > 0.0
        else _empty_set()
    )
    # Same over-fetch discipline as the single-anchor flow — when hard
    # filters are active, raise the LIMIT so the post-filter pool stays
    # usefully sized.
    effective_quality_limit = (
        quality_limit * 2 if metadata_filters is not None else quality_limit
    )
    quality_task = (
        _traced_lane_fetch(
            "quality",
            {"bucket": repeated_quality_bucket, "limit": effective_quality_limit},
            fetch_similarity_quality_candidates(
                bucket=repeated_quality_bucket,
                limit=effective_quality_limit,
                metadata_filters=metadata_filters,
            ),
        )
        if repeated_quality_bucket is not None and cohesion_by_lane["quality"] > 0.0
        else _empty_set()
    )
    anchor_cast_task = fetch_similarity_top_billed_cast(anchor_ids)
    anchor_award_task = fetch_similarity_award_category_tags(anchor_ids)

    # V3.1 themes recall (multi-anchor): we need anchor-side themes
    # IDFs *before* building candidate_ids so consensus traits can be
    # computed and the recall query can fire in this same first
    # gather. Compute themes_union here (themes_trait_sets already
    # built above) and fetch its IDFs in parallel.
    themes_union_for_recall: set[tuple[int, int]] = set()
    for trait_set in themes_trait_sets:
        themes_union_for_recall |= trait_set
    themes_recall_idf_task = (
        fetch_trait_idfs(list(themes_union_for_recall))
        if themes_union_for_recall
        else _empty_idf_dict()
    )

    (
        (shape_scores, vector_space_weights, vector_space_cohesion, mean_pairwise_cosine),
        (auteur_term_ids, has_auteur_anchor, director_candidate_terms),
        franchise_candidate_ids,
        studio_candidate_ids,
        source_candidate_ids,
        quality_candidate_ids,
        anchor_cast_by_movie,
        anchor_award_by_movie,
        themes_recall_idfs,
    ) = await asyncio.gather(
        shape_task,
        director_combo_task,
        franchise_task,
        studio_task,
        source_task,
        quality_task,
        anchor_cast_task,
        anchor_award_task,
        themes_recall_idf_task,
    )

    # Multi-anchor themes recall: build consensus pool with the
    # cohesion-IDF tradeoff, then collapse to the same single-anchor
    # SQL helper.
    themes_recall_consensus = _multi_anchor_consensus_themes_traits(
        themes_trait_sets,
        themes_recall_idfs,
        n,
    )
    themes_recall_candidate_ids: set[int] = (
        await _traced_lane_fetch(
            "themes_recall",
            _themes_recall_match(themes_recall_consensus),
            fetch_movie_ids_by_themes_recall(
                keyword_ids=[
                    tid for (kind, tid) in themes_recall_consensus
                    if kind == TRAIT_KIND_OVERALL_KEYWORD
                ],
                concept_tag_ids=[
                    tid for (kind, tid) in themes_recall_consensus
                    if kind == TRAIT_KIND_CONCEPT_TAG
                ],
                genre_ids=[
                    tid for (kind, tid) in themes_recall_consensus
                    if kind == TRAIT_KIND_TMDB_GENRE
                ],
                metadata_filters=metadata_filters,
            ),
        )
        if themes_recall_consensus
        else set()
    )

    # Cast / specific_award cohesion needs the just-fetched anchor data.
    cast_trait_sets = [
        anchor_cast_by_movie.get(anchor_id, set()) for anchor_id in anchor_ids
    ]
    specific_award_trait_sets = [
        anchor_award_by_movie.get(anchor_id, set()) for anchor_id in anchor_ids
    ]
    cohesion_by_lane["cast"] = _metadata_cohesion(cast_trait_sets)
    cohesion_by_lane["specific_award"] = _multi_anchor_specific_award_cohesion(
        specific_award_trait_sets, anchor_count=n
    )

    # V2 low-cohesion fallback: when both vector cohesion and every
    # metadata lane's cohesion are weak, the centroid lands in noise. Bail
    # out to round-robin per-anchor single-anchor results.
    metadata_max_cohesion = max(cohesion_by_lane.values(), default=0.0)
    if (
        mean_pairwise_cosine < LOW_COHESION_VECTOR_THRESHOLD
        and metadata_max_cohesion < LOW_COHESION_METADATA_MAX_THRESHOLD
    ):
        return await _low_cohesion_fallback(
            anchor_ids,
            anchor_rows,
            vectors_by_anchor,
            studio_entries_by_company_id,
            director_terms_by_anchor,
            limit=limit,
            qdrant_limit=qdrant_limit,
            quality_limit=quality_limit,
            mean_pairwise_cosine=mean_pairwise_cosine,
            metadata_filters=metadata_filters,
        )

    # V3 raw lane weights. Shape scales with mean cohesion (range
    # [0.36, 1.20]); proportional metadata lanes scale by their cohesion
    # factor. Director is passthrough (raw = absolute contribution).
    # V3.4.4: rare_keyword is no longer a lane (removed entirely).
    #
    # V3 §4.1 quality bucket-conditional weight: the middle bucket gets
    # half the base weight (0.03 vs 0.06) — V2's 0.06 was over-firing
    # for prestige-by-2-of-3 cases (Lawrence of Arabia / Citizen Kane
    # surfacing for Godfather despite no shape adjacency).
    shape_raw = _shape_raw_for_multi_anchor(mean_pairwise_cosine)
    quality_base = (
        0.03 if repeated_quality_bucket not in {"cult_garbage", "prestige"} else 0.06
    )
    raw_lane_weights: dict[LaneName, float] = {
        "shape": shape_raw,
        "director": 1.00,                               # passthrough; raw is absolute
        "franchise": 0.12 * cohesion_by_lane["franchise"],
        "studio": 0.06 * cohesion_by_lane["studio"],   # debug-only weight
        "source": 0.04 * cohesion_by_lane["source"],
        "quality": quality_base * cohesion_by_lane["quality"],
        "format": 0.04 * cohesion_by_lane["format"],
        "themes": BASE_LANE_WEIGHTS["themes"] * cohesion_by_lane["themes"],
        # Cast weight is finalized after the V3 cast lane runs (it
        # encodes cohesion explicitly via 0.05 + 0.10 * ratio).
        "cast": 0.0,
        "specific_award": 0.04 * cohesion_by_lane["specific_award"],
    }
    # `_normalize_weights` is deferred to after cast scoring — see below.

    candidate_ids = set(shape_scores)
    candidate_ids.update(director_candidate_terms)
    candidate_ids.update(franchise_candidate_ids)
    candidate_ids.update(studio_candidate_ids)
    candidate_ids.update(source_candidate_ids)
    candidate_ids.update(quality_candidate_ids)
    candidate_ids.update(themes_recall_candidate_ids)  # V3.1 themes recall path
    candidate_ids -= set(anchor_ids)

    # Retrieval-side pool sizes for the branch span. Multi-anchor lanes gate on
    # cohort agreement (the same predicates guarding each `_empty_set()` above),
    # so keying on those predicates records a lane only when its query actually
    # ran — fired-but-empty present at 0, gated-off absent. shape always runs.
    retrieval_counts: dict[str, int] = {"shape": len(shape_scores)}
    if cohesion_by_lane["director"] > 0.0 or has_auteur_anchor:
        retrieval_counts["director"] = len(director_candidate_terms)
    if cohesion_by_lane["franchise"] > 0.0:
        retrieval_counts["franchise"] = len(franchise_candidate_ids)
    if cohesion_by_lane["studio"] > 0.0:
        retrieval_counts["studio"] = len(studio_candidate_ids)
    if cohesion_by_lane["source"] > 0.0:
        retrieval_counts["source"] = len(source_candidate_ids)
    if repeated_quality_bucket is not None and cohesion_by_lane["quality"] > 0.0:
        retrieval_counts["quality"] = len(quality_candidate_ids)
    if themes_recall_consensus:
        retrieval_counts["themes_recall"] = len(themes_recall_candidate_ids)
    retrieval_total = len(candidate_ids)

    # Candidate-side reads.
    candidate_rows_task = fetch_similarity_signal_rows(list(candidate_ids))
    candidate_cast_task = (
        fetch_similarity_top_billed_cast(list(candidate_ids))
        if cohesion_by_lane["cast"] > 0.0
        else _empty_movie_set_dict()
    )
    candidate_award_tags_task = (
        fetch_similarity_award_category_tags(list(candidate_ids))
        if cohesion_by_lane["specific_award"] > 0.0
        else _empty_movie_set_dict()
    )
    # V3.3.2: always fetch award signals for both anchors and candidates —
    # the shape classifier needs them regardless of repeated_quality_bucket.
    # Anchors' award signals also drive the cohort's shape composition.
    award_signals_task = fetch_similarity_award_signals(
        list(candidate_ids) + list(anchor_ids)
    )
    # Source IDFs across the union of source types touched by anchors.
    source_idf_pairs_task = (
        fetch_trait_idfs([(TRAIT_KIND_SOURCE_MATERIAL, t) for t in source_ids])
        if source_ids and cohesion_by_lane["source"] > 0.0
        else _empty_idf_dict()
    )
    # V3 §2.6: rare-keyword scores against the UNION of anchor traits
    # (any anchor's rare keyword is a valid signal — rare-trait IDFs
    # don't compound across anchors). Themes scores against the
    # REPEATED subset only, which is a subset of the union. So we
    # fetch IDFs for the full union once and feed both lanes.
    themes_repeated = _multi_anchor_themes_repeated(themes_trait_sets)
    themes_union: set[tuple[int, int]] = set()
    for trait_set in themes_trait_sets:
        themes_union |= trait_set
    # Always fetch the union when any anchor has trait pool entries —
    # rare-keyword needs the IDFs even when themes cohesion is zero.
    # Themes scoring still no-ops when `themes_repeated` is empty
    # because it iterates over `themes_repeated` directly.
    themes_idf_pairs_task = (
        fetch_trait_idfs(list(themes_union))
        if themes_union
        else _empty_idf_dict()
    )

    (
        candidate_rows,
        candidate_cast_by_movie,
        candidate_award_tags_by_movie,
        award_signals,
        source_idf_pairs,
        themes_idf_pairs,
    ) = await asyncio.gather(
        candidate_rows_task,
        candidate_cast_task,
        candidate_award_tags_task,
        award_signals_task,
        source_idf_pairs_task,
        themes_idf_pairs_task,
    )

    source_idfs: dict[int, float] = {
        trait_id: idf for (_, trait_id), idf in source_idf_pairs.items()
    }
    # Themes IDFs stay keyed by (kind, trait_id) so colliding numeric IDs
    # across the keyword/concept/genre families don't cross-contaminate.
    themes_idfs: dict[tuple[int, int], float] = themes_idf_pairs

    # One-pass parse of frequently-reused row derivations (see
    # _parse_candidate_rows). Downstream orchestrator-level lookups
    # read from this dict instead of re-parsing each row 2-3 times.
    parsed_by_movie = _parse_candidate_rows(candidate_rows)

    # ----- Per-lane scoring -----
    # V3 §2.1 director: per-shared-director absolute contributions
    # (curated 0.20-0.30 vs cohesion-only ≤0.10), max-over-d. Also
    # returns max_ratio per candidate for the §2.1 cohesion floor.
    director_scores, director_max_ratio = _multi_anchor_director_score(
        candidate_terms=director_candidate_terms,
        anchor_director_sets=director_trait_sets,
        auteur_term_ids=auteur_term_ids,
        n=n,
    )

    franchise_candidate_traits = {
        mid: set().union(*_franchise_traits(row))
        for mid, row in candidate_rows.items()
        if mid in franchise_candidate_ids
    }
    franchise_scores = _score_multi_trait_count(
        candidate_terms=franchise_candidate_traits,
        anchor_trait_sets=franchise_trait_sets,
        anchor_count=n,
    )

    studio_candidate_traits = {
        mid: {
            entry.company_id
            for entry in _active_studio_entries(row, studio_entries_by_company_id)
        }
        for mid, row in candidate_rows.items()
        if mid in studio_candidate_ids
    }
    # Studio is multiplicative in V2 — but multi-anchor still uses
    # repetition-count scoring as input to that multiplier so a candidate
    # matching every anchor's studio gets the full +10% boost while a
    # 1-of-3 match gets ~3.3%.
    studio_scores = _score_multi_trait_count(
        candidate_terms=studio_candidate_traits,
        anchor_trait_sets=studio_trait_sets,
        anchor_count=n,
    )

    # Source: V2 IDF-weighted per-anchor match. Common types ("novel")
    # contribute ~0.20 even at full anchor coverage; rare types ride at
    # nearly 1.0.
    source_candidate_traits = {
        mid: _as_int_set(row.get("source_material_type_ids"))
        for mid, row in candidate_rows.items()
        if mid in source_candidate_ids
    }
    source_scores = _score_multi_trait_count(
        candidate_terms=source_candidate_traits,
        anchor_trait_sets=source_trait_sets,
        anchor_count=n,
        weight_fn=lambda shared: max(
            (source_idfs.get(t, 0.0) for t in shared), default=0.0
        ),
    )

    # Quality: repetition-driven scoring — a candidate in the same bucket
    # as the consensus gets full match strength; cross-bucket candidates
    # use the V2 per-bucket formula scaled by the consensus repetition.
    quality_scores: dict[int, float] = {}
    if repeated_quality_bucket is not None and cohesion_by_lane["quality"] > 0.0:
        repeated_anchor_count = sum(
            1 for traits in quality_trait_sets if repeated_quality_bucket in traits
        )
        repetition_ratio = repeated_anchor_count / n
        for mid, row in candidate_rows.items():
            if mid not in quality_candidate_ids:
                continue
            if _quality_bucket(row) == repeated_quality_bucket:
                quality_scores[mid] = repetition_ratio
            else:
                quality_scores[mid] = (
                    _quality_score_v2(
                        repeated_quality_bucket, row, award_signals.get(mid)
                    )
                    * repetition_ratio
                )

    # Format: same-bucket-or-not vs the repeated bucket, every candidate.
    # Uses parsed_by_movie so candidate_format_bucket_by_movie below
    # shares the same format_bucket evaluation (one call per candidate).
    format_scores: dict[int, float] = {}
    if repeated_format_bucket is not None and cohesion_by_lane["format"] > 0.0:
        for mid, parsed in parsed_by_movie.items():
            if parsed.format_bucket_value == repeated_format_bucket:
                format_scores[mid] = 1.0

    # Themes: per-candidate share of repeated-trait IDF mass.
    themes_scores = _multi_anchor_themes_scores(
        candidate_rows=candidate_rows,
        repeated_traits=themes_repeated,
        idf_lookup=themes_idfs,
    )

    # V3 §2.5 cast: generic N-anchor bucket-with-floor on shared
    # top-3 billed leads. Returns per-candidate score, per-candidate
    # floor, and the cohesion-driven lane weight (0.05 + 0.10 * ratio).
    (
        cast_scores,
        cast_floor_by_movie,
        cast_lane_weight,
    ) = _multi_anchor_cast_v3(
        candidate_top_billing=candidate_cast_by_movie,
        anchor_top_billing_sets=cast_trait_sets,
        n=n,
    )
    raw_lane_weights["cast"] = cast_lane_weight

    # V3.4.4: rare_keyword lane removed entirely (see single-anchor flow).

    # Now finalize the lane weights with the cast lane's cohesion-
    # derived value. Weights downstream of this point use `lane_weights`.
    lane_weights = _normalize_weights(raw_lane_weights)

    # Specific award: tier-weighted score over repeated tags.
    specific_award_scores = _multi_anchor_specific_award_scores(
        candidate_award_tags_by_movie=candidate_award_tags_by_movie,
        anchor_trait_sets=specific_award_trait_sets,
        anchor_count=n,
    )

    # Country/language consensus multiplier per candidate. Uses
    # parsed_by_movie so we don't re-run country_set() per row.
    country_consensus_match: dict[int, bool] = {}
    if consensus_countries:
        for mid, parsed in parsed_by_movie.items():
            country_consensus_match[mid] = bool(
                parsed.country_tags & consensus_countries
            )

    # Per the V2 spec, multi-anchor medium handling is left to a future
    # iteration — multi-anchor candidates are not scored or filtered by
    # medium in V2.0. Pass an empty multiplier dict so build_results
    # leaves combined_score untouched.

    # V3 §3.1 shorts dominance: when ≥50% of anchors are shorts, the
    # user is signaling they want shorts. Disable the harsh downrank
    # in that case (the lane caller passes apply_shorts_downrank=False).
    short_anchor_count = sum(
        1
        for traits in format_trait_sets
        if "short" in traits
    )
    shorts_dominant = (
        short_anchor_count / max(1, n) >= SHORTS_MULTI_ANCHOR_DOMINANCE_THRESHOLD
    )
    candidate_format_bucket_by_movie: dict[int, FormatBucket] = {
        mid: format_bucket(row.get("keyword_ids") or ())
        for mid, row in candidate_rows.items()
    }

    # V3.3 shape multiplier — multi-anchor: build per-shape cohesion
    # (M_s/N) across the cohort. V3.3.2 award-aware classification
    # consumes per-movie award signals; cross-bucket strengths inside
    # `_shape_multiplier` let mixed cohorts apply partial boosts on
    # cross-shape candidates. Threshold gating happens inside the
    # multiplier helper.
    anchor_shape_counts: dict[str, int] = {}
    for anchor_id_for_shape in anchor_ids:
        anchor_s = _classify_shape(
            anchor_rows[anchor_id_for_shape],
            award_signals.get(anchor_id_for_shape),
        )
        if anchor_s is not None:
            anchor_shape_counts[anchor_s] = (
                anchor_shape_counts.get(anchor_s, 0) + 1
            )
    anchor_shape_cohesion: dict[str, float] = {
        shape: count / float(n) for shape, count in anchor_shape_counts.items()
    }
    candidate_shape_by_movie: dict[int, str | None] = {
        movie_id: _classify_shape(row, award_signals.get(movie_id))
        for movie_id, row in candidate_rows.items()
    }

    lane_scores: dict[LaneName, dict[int, float]] = {
        "shape": shape_scores,
        "director": {mid: s for mid, s in director_scores.items() if s > 0.0},
        "franchise": {mid: s for mid, s in franchise_scores.items() if s > 0.0},
        "studio": {},   # debug-only; multiplier handles the contribution
        "source": {mid: s for mid, s in source_scores.items() if s > 0.0},
        "quality": {mid: s for mid, s in quality_scores.items() if s > 0.0},
        "format": format_scores,
        "themes": {mid: s for mid, s in themes_scores.items() if s > 0.0},
        "cast": {mid: s for mid, s in cast_scores.items() if s > 0.0},
        "specific_award": {
            mid: s for mid, s in specific_award_scores.items() if s > 0.0
        },
    }

    # V3.4 bucket-weaver inputs for the multi-anchor flow. Includes the
    # lead-actor bucket (single-anchor's silent counterpart) when cohort
    # cast cohesion clears the gate. Memberships are derived from the
    # same per-lane data already computed above, no extra DB reads.
    bucket_signals, bucket_memberships = _compute_multi_anchor_bucket_data(
        anchor_ids=anchor_ids,
        anchor_rows=anchor_rows,
        director_trait_sets=director_trait_sets,
        auteur_term_ids=auteur_term_ids,
        themes_trait_sets=themes_trait_sets,
        themes_idfs=themes_idfs,
        candidate_rows=candidate_rows,
        director_candidate_terms=director_candidate_terms,
        franchise_candidate_ids=franchise_candidate_ids,
        candidate_top_billing=candidate_cast_by_movie,
        cast_trait_sets=cast_trait_sets,
    )

    ranked, counts = _build_results(
        anchor_ids=anchor_ids,
        lane_scores=lane_scores,
        lane_weights=lane_weights,
        limit=limit,
        studio_score_by_movie=studio_scores,
        country_consensus_match_by_movie=country_consensus_match,
        cast_floor_by_movie=cast_floor_by_movie,
        director_floor_max_ratio_by_movie=director_max_ratio,
        candidate_format_bucket_by_movie=candidate_format_bucket_by_movie,
        anchor_active_format_bucket=(
            "short" if shorts_dominant else repeated_format_bucket
        ),
        apply_shorts_downrank=not shorts_dominant,
        apply_shorts_boost=shorts_dominant,
        anchor_format_bucket=repeated_format_bucket,
        enforce_format_top_lock=repeated_format_bucket is not None,
        anchor_shape_cohesion=anchor_shape_cohesion,
        candidate_shape_by_movie=candidate_shape_by_movie,
        bucket_signals=bucket_signals,
        bucket_memberships_by_movie=bucket_memberships,
    )

    active_anchor_types: list[AnchorType] = ["standard_shape"]
    if cohesion_by_lane["franchise"] > 0.0:
        active_anchor_types.append("franchise_dominant")
    if cohesion_by_lane["studio"] > 0.0:
        active_anchor_types.append("studio_lineage")
    if cohesion_by_lane["source"] > 0.0:
        active_anchor_types.append("source_material")
    if repeated_quality_bucket in {"cult_garbage", "prestige"}:
        active_anchor_types.append(repeated_quality_bucket)  # type: ignore[arg-type]
    # director_signature parity with single-anchor: fire when any anchor's
    # director is on the curated auteur list. `has_auteur_anchor` mirrors the
    # single-anchor `anchor_directors & auteur_term_ids` gate over the union of
    # anchor directors. Without this the multi-anchor branch silently omitted
    # director_signature from active_anchor_types even for all-auteur cohorts,
    # so the traced attribute meant different things single vs multi.
    if has_auteur_anchor:
        active_anchor_types.append("director_signature")

    # Per-anchor active-anchor-types — useful for debugging centroid drift
    # cases (Best Picture trio where Schindler/12YS dominated and pushed
    # The Godfather adjacents off the page).
    per_anchor_active = _per_anchor_active_anchor_types(
        anchor_ids,
        anchor_rows,
        director_terms_by_anchor,
        anchor_studio_entries,
    )

    return SimilarMoviesSearchResult(
        anchor_movie_ids=anchor_ids,
        ranked=ranked,
        active_anchor_types=active_anchor_types,
        debug=SimilarMoviesDebug(
            vector_space_weights=vector_space_weights,
            vector_space_cohesion=vector_space_cohesion,
            raw_lane_weights=raw_lane_weights,
            normalized_lane_weights=lane_weights,
            candidate_counts_by_lane=counts,
            retrieval_counts_by_lane=retrieval_counts,
            retrieval_total=retrieval_total,
            lane_cohesion=dict(cohesion_by_lane),
            anchor_format_bucket=repeated_format_bucket,
            consensus_countries=sorted(consensus_countries, key=str),
            per_anchor_active_anchor_types=per_anchor_active,
            shorts_dominant=shorts_dominant,
            anchor_shape_cohesion=anchor_shape_cohesion,
        ),
    )


# ---------------------------------------------------------------------------
# V3 lane helpers
# ---------------------------------------------------------------------------


def _country_consensus_per_candidate(
    *,
    anchor_country_tags: frozenset[int | str],
    candidate_rows: dict[int, dict],
) -> dict[int, bool]:
    """V3 §2.4 single-anchor country/language consensus per candidate.

    Anchor's own country tags become the consensus set. Candidates with
    ≥1 country in common are "match" → +5%; candidates that DO carry
    country tags but none in common are "mismatch" → -25%; candidates
    with no country tags get neither (preserves V2 "we can't tell").

    Returns ``{movie_id: bool}`` where True = match, False = mismatch,
    absent = no signal. ``_build_results`` reads the same shape that
    multi-anchor uses.
    """
    out: dict[int, bool] = {}
    if not anchor_country_tags:
        return out
    for mid, row in candidate_rows.items():
        candidate_countries = country_set(row.get("keyword_ids") or ())
        if not candidate_countries:
            continue
        out[mid] = bool(candidate_countries & anchor_country_tags)
    return out


# Generic IDF tier boundaries. Same numeric values as the old
# rare_keyword tiers — these delineate "common" / "moderate" / "rare"
# by IDF independent of any specific lane's use. 0.30 separates
# trivial-overlap tags (DRAMA, COMEDY) from moderately-distinctive
# ones; 0.55 separates moderate from genuinely-rare (the bucket's
# own threshold).
IDF_TIER_LOW_MAX = 0.30        # idf < 0.30 → trivial tier
IDF_TIER_MODERATE_MAX = 0.55   # 0.30 ≤ idf < 0.55 → moderate; idf ≥ 0.55 → rare


# Themes-lane minimum normalizer denominator. Anchors with a small
# trait pool (e.g., Barbie at IDF mass 2.5) give very small denominators;
# this floor prevents any single high-IDF trait from monopolizing the
# score by dividing by something smaller than itself.
THEMES_MIN_DENOMINATOR = 1.0

# V3.4.5: themes-lane compounding multiplier. Rewards aggregate co-
# occurrence of moderate-or-rarer tags by *multiplying* the candidate's
# themes lane score, not adding a flat bonus. The multiplicative shape
# is what makes the boost actually compound with quantity — additive
# 0.15 cap (V3.4.4) was too modest to recover candidates like Matrix's
# Tron/Blade Runner that share multiple moderate-tier sci-fi tags.
#
# Why multiplying the lane score is safe (unlike the V3.4.3 multiplier
# which multiplied the *final* score and caused tail dominance): the
# themes lane is proportional, weighted at ~0.10 in the additive sum.
# A 2× lane score multiplier translates to ~+0.05 final-score lift, not
# runaway amplification. The boost decays naturally with rank because
# the themes lane itself decays.
#
# Random-pair sampling showed p99 of shared moderate+high IDF = 0.000
# (324 random pairs, zero crossed 0.5), so engaging at threshold 0.30
# (one shared moderate trait) means anything beyond a single match
# starts compounding.
THEMES_COMBO_MIN_IDF   = IDF_TIER_LOW_MAX  # only moderate+ tags compound
THEMES_COMBO_THRESHOLD = 0.30      # sum of qualifying IDFs to engage
THEMES_COMBO_FACTOR    = 0.50      # multiplier strength: mult = 1 + factor*excess


def _themes_combo_multiplier(
    shared: set[tuple[int, int]],
    idf_lookup: dict[tuple[int, int], float],
) -> float:
    """V3.4.5: compounding *multiplier* on themes lane scores.

    Rewards aggregate co-occurrence of moderate-or-rarer shared tags.
    Returns the multiplier to apply to the candidate's base themes
    score (1.0 means no boost). Shape:

        sum_qualifying = sum(idf for shared if idf >= MIN_IDF)
        excess = max(0, sum_qualifying - THRESHOLD)
        mult = 1.0 + FACTOR * excess

    Multiplicative replacement for V3.4.4's additive bonus — same
    qualifier (shared moderate+high IDF sum) but compounding shape
    rather than threshold + linear-up-to-cap.
    """
    sum_qualifying = sum(
        idf for trait in shared
        for idf in [idf_lookup.get(trait, 0.0)]
        if idf >= THEMES_COMBO_MIN_IDF
    )
    excess = sum_qualifying - THEMES_COMBO_THRESHOLD
    if excess <= 0.0:
        return 1.0
    return 1.0 + THEMES_COMBO_FACTOR * excess


def _single_anchor_themes_scores(
    *,
    anchor_traits: set[tuple[int, int]],
    candidate_rows: dict[int, dict],
    idf_lookup: dict[tuple[int, int], float],
) -> dict[int, float]:
    """V3 §2.3 single-anchor themes lane.

    For each candidate, score = sum(IDF for shared traits) / max(
    THEMES_MIN_DENOMINATOR, sum(IDF for anchor traits)). Same trait
    pool as multi-anchor themes (concept_tags + non-registry overall
    keywords + genres, minus country/medium/format).

    Per pre-flight #3, anchors with a small IDF mass (e.g., Barbie at
    ~2.5) score top candidates well above 0.05 because partial overlap
    over a small denominator gives meaningful values; the
    ``THEMES_MIN_DENOMINATOR`` floor keeps the result in [0, 1].
    """
    if not anchor_traits:
        return {}
    anchor_idf_sum = sum(idf_lookup.get(t, 0.0) for t in anchor_traits)
    denom = max(THEMES_MIN_DENOMINATOR, anchor_idf_sum)
    scores: dict[int, float] = {}
    for mid, row in candidate_rows.items():
        candidate_traits = _themes_traits_for_movie(row)
        shared = anchor_traits & candidate_traits
        if not shared:
            continue
        shared_idf = sum(idf_lookup.get(t, 0.0) for t in shared)
        if shared_idf > 0.0:
            base = shared_idf / denom
            # V3.4.5 compounding multiplier on aggregate moderate+ overlap.
            mult = _themes_combo_multiplier(shared, idf_lookup)
            scores[mid] = _clamp(base * mult)
    return scores


# V3.4.4: the rare_keyword lane (V3 §2.6) is removed entirely. Tag
# overlap is fully scored by the themes lane (with a compounding bonus
# migrated from the old combo signal). Rare-match exploration runs
# through the rare_keyword bucket in the greedy weaver, gated by
# WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN. The IDF_TIER_* boundaries used
# by the themes-recall consensus logic are defined earlier in the
# module (alongside themes constants).
RARE_KW_BUCKET_SIGNAL_SCALE = 1.00  # idf-sum that maps to bucket signal 1.0


# V3 §2.5 cast bucket-with-floor parameters.
CAST_LANE_WEIGHT_BASE = 0.05       # silent component — pooled into 0.05 weight
CAST_LANE_WEIGHT_RATIO = 0.10      # max weight = 0.05 + 0.10 = 0.15 at full cohesion
CAST_FLOOR_RATIO_THRESHOLD = 0.5
CAST_FLOOR_BASE = 0.25
CAST_FLOOR_RATIO_INC = 0.20        # max floor = 0.25 + 0.20 = 0.45
CAST_FLOOR_SHAPE_GATE = 0.30


def _multi_anchor_cast_v3(
    *,
    candidate_top_billing: dict[int, set[int]],
    anchor_top_billing_sets: list[set[int]],
    n: int,
) -> tuple[dict[int, float], dict[int, float], float]:
    """V3 §2.5 cast scoring with bucket-with-floor.

    Returns ``(per_candidate_lane_score, per_candidate_floor, lane_weight)``.

    ``M_a`` = number of anchors with actor a in top-3 billing.
    ``shared_leads`` = {actors with M_a >= 2}.
    ``max_M`` = max(M_a). ``ratio = max_M / N``.

    Per candidate:
      - matches = |candidate.top_3_billing ∩ shared_leads|
      - lane_score = matches / |shared_leads|     (in [0,1])
      - floor = 0.25 + 0.20 * ratio if ratio ≥ 0.5 AND candidate matched ≥1 lead

    The lane_weight is shared across candidates (a function of cohesion
    only, not the candidate). It's returned alongside the per-candidate
    dicts so the caller can pass it through `_normalize_weights`.
    """
    # No shared leads → lane is silent. Return weight 0.0 (not the
    # CAST_LANE_WEIGHT_BASE) so the lane doesn't dilute other lanes'
    # normalized weights via the `_normalize_weights` denominator. V2
    # behavior was strict gating; V3 keeps that behavior in the
    # no-cohesion case.
    if n <= 0 or not anchor_top_billing_sets:
        return {}, {}, 0.0
    # Compute M_a for each actor.
    m_per_actor: dict[int, int] = {}
    for billing in anchor_top_billing_sets:
        for actor in billing:
            m_per_actor[actor] = m_per_actor.get(actor, 0) + 1
    shared_leads = {a for a, m in m_per_actor.items() if m >= 2}
    if not shared_leads:
        return {}, {}, 0.0
    max_m = max(m_per_actor[a] for a in shared_leads)
    ratio = max_m / n
    lane_weight = CAST_LANE_WEIGHT_BASE + CAST_LANE_WEIGHT_RATIO * ratio
    floor_value = (
        CAST_FLOOR_BASE + CAST_FLOOR_RATIO_INC * ratio
        if ratio >= CAST_FLOOR_RATIO_THRESHOLD
        else 0.0
    )
    lane_scores: dict[int, float] = {}
    floor_by_movie: dict[int, float] = {}
    for mid, billing in candidate_top_billing.items():
        if not billing:
            continue
        matches = len(billing & shared_leads)
        if matches == 0:
            continue
        lane_scores[mid] = _clamp(matches / len(shared_leads))
        if floor_value > 0.0:
            floor_by_movie[mid] = floor_value
    return lane_scores, floor_by_movie, lane_weight


# V3 §3.1 shorts harsh downrank.
SHORTS_DOWNRANK_MULTIPLIER = 0.30
SHORTS_TOP_SECTION_MAX = 1          # max shorts in top 10 when anchor isn't a short
SHORTS_MULTI_ANCHOR_BOOST = 1.10    # multi-anchor with shorts dominance — boost shorts
SHORTS_MULTI_ANCHOR_DOMINANCE_THRESHOLD = 0.5  # ratio of short anchors to fire boost


def _score_multi_trait_count(
    *,
    candidate_terms: dict[int, set[int] | set[str]],
    anchor_trait_sets: list[set[int] | set[str]],
    anchor_count: int,
    weight_fn=None,
) -> dict[int, float]:
    """Per-anchor-match scoring shared across the multi-anchor lanes.

    Default behavior: each anchor whose trait set intersects the candidate
    contributes ``1.0`` to the numerator; final score is averaged over
    ``anchor_count`` so a candidate matching every anchor scores 1.0. If
    ``weight_fn`` is supplied, each matching anchor contributes
    ``weight_fn(shared_traits)`` instead — used by the V2 multi-anchor
    source lane to weight per-anchor matches by the IDF of the shared
    type, so common-tag overlaps don't ride at full strength.
    """
    scores: dict[int, float] = {}
    for movie_id, traits in candidate_terms.items():
        if not traits:
            continue
        total = 0.0
        for anchor_traits in anchor_trait_sets:
            shared = traits & anchor_traits
            if not shared:
                continue
            total += weight_fn(shared) if weight_fn is not None else 1.0
        if total > 0.0:
            scores[movie_id] = _clamp(total / anchor_count)
    return scores


# Precomputed per-candidate derivations of frequently-reused row
# fields. The single-anchor flow alone hits ``keyword_ids`` three times
# per row (format_bucket, _medium_tags_for_movie, country_set) and the
# franchise pool check parses lineage + universe arrays a second time
# after the lane scoring already did the same. Building these once and
# reading them from a dict downstream cuts repeated ``_as_int_set`` /
# ``format_bucket`` work on every candidate.
#
# Scope is deliberately narrow — only the orchestrator-level call sites
# consume from this slice. The per-lane scoring helpers (e.g.
# ``_franchise_score_v2``, ``_quality_score_v2``) still accept ``row:
# dict`` so the diff stays small and the scoring contract unchanged.
@dataclass(frozen=True, slots=True)
class _ParsedRowSlice:
    format_bucket_value: FormatBucket
    medium_tags: frozenset[int]
    country_tags: frozenset[int | str]
    themes_traits: frozenset[tuple[int, int]]
    franchise_pool: frozenset[int]   # lineage_entry_ids | shared_universe_entry_ids


def _parse_candidate_rows(rows: dict[int, dict]) -> dict[int, _ParsedRowSlice]:
    """One-pass extraction of the orchestrator's frequently-reused row
    derivations. Each value is computed exactly once per candidate.
    """
    out: dict[int, _ParsedRowSlice] = {}
    for movie_id, row in rows.items():
        keyword_ids_set = _as_int_set(row.get("keyword_ids"))
        keyword_ids_tuple = row.get("keyword_ids") or ()
        out[movie_id] = _ParsedRowSlice(
            format_bucket_value=format_bucket(keyword_ids_tuple),
            medium_tags=frozenset(keyword_ids_set & MEDIUM_TAG_IDS),
            country_tags=frozenset(country_set(keyword_ids_tuple)),
            themes_traits=frozenset(_themes_traits_for_movie(row)),
            franchise_pool=frozenset(
                _as_int_set(row.get("lineage_entry_ids"))
                | _as_int_set(row.get("shared_universe_entry_ids"))
            ),
        )
    return out


def _themes_traits_for_movie(row: dict) -> set[tuple[int, int]]:
    """Build the themes-lane trait pool for a movie row.

    Combines keyword_ids, concept_tag_ids, and genre_ids minus the tags
    that are already covered by other V2 lanes (country/language tags
    feed the coherence multiplier; medium tags feed the medium
    multiplier; format tags feed the format lane). Returns ``(kind,
    trait_id)`` tuples — the three families share numeric IDs (e.g.
    ``Genre.HORROR.genre_id = 14`` collides with
    ``OverallKeyword.BASEBALL.keyword_id = 14``), so the kind has to be
    on the trait key or repeated-trait counts and IDF lookups land on
    the wrong family.
    """
    keyword_ids = _as_int_set(row.get("keyword_ids"))
    excluded = (
        COUNTRY_LANGUAGE_KEYWORD_IDS
        | MEDIUM_TAG_IDS
        | FORMAT_KEYWORD_IDS_ALL
    )
    keyword_ids -= excluded
    out: set[tuple[int, int]] = set()
    for k in keyword_ids:
        out.add((TRAIT_KIND_OVERALL_KEYWORD, k))
    for c in _as_int_set(row.get("concept_tag_ids")):
        out.add((TRAIT_KIND_CONCEPT_TAG, c))
    for g in _as_int_set(row.get("genre_ids")):
        out.add((TRAIT_KIND_TMDB_GENRE, g))
    return out


def _repeated_quality_bucket(
    quality_trait_sets: list[set[str]],
) -> str | None:
    counts: dict[str, int] = {}
    for traits in quality_trait_sets:
        for trait in traits:
            counts[trait] = counts.get(trait, 0) + 1
    repeated = [(count, bucket) for bucket, count in counts.items() if count >= 2]
    if not repeated:
        return None
    repeated.sort(reverse=True)
    return repeated[0][1]


def _repeated_format_bucket(
    format_trait_sets: list[set[str]],
) -> FormatBucket | None:
    """Return the format bucket repeated by ≥2 anchors, or None.

    When more than one bucket repeats (uncommon but possible — e.g. two
    docs and two narrative features in a 4-anchor set), returns the most
    repeated. Ties broken alphabetically for stable behavior.
    """
    counts: dict[str, int] = {}
    for traits in format_trait_sets:
        for trait in traits:
            counts[trait] = counts.get(trait, 0) + 1
    repeated = sorted(
        ((count, bucket) for bucket, count in counts.items() if count >= 2),
        reverse=True,
    )
    if not repeated:
        return None
    return repeated[0][1]  # type: ignore[return-value]


def _consensus_country_set(
    country_trait_sets: list[frozenset[int | str]],
) -> set[int | str]:
    """Return the country/language buckets shared by ≥2 anchors."""
    counts: dict[int | str, int] = {}
    for traits in country_trait_sets:
        for trait in traits:
            counts[trait] = counts.get(trait, 0) + 1
    return {bucket for bucket, count in counts.items() if count >= 2}


def _multi_anchor_specific_award_cohesion(
    award_trait_sets: list[set[int]],
    *,
    anchor_count: int,
) -> float:
    """V2 specific-award lane cohesion: log-curve scaled by tier specificity.

    The lane fires only when ≥2 anchors share a category_tag_id. The
    specificity factor discounts cohesion when the only repetition is at
    a coarser level — three different acting awards repeating only at L1
    LEAD_ACTING get partial cohesion; three disjoint groups repeating at
    L2 get a mild one. A perfect L0 BEST_PICTURE three-way match maxes
    out at 2.0.
    """
    if anchor_count < 2 or not award_trait_sets:
        return 0.0
    counts: dict[int, int] = {}
    for traits in award_trait_sets:
        for tag_id in traits:
            counts[tag_id] = counts.get(tag_id, 0) + 1
    repeated = {tag_id: c for tag_id, c in counts.items() if c >= 2}
    if not repeated:
        return 0.0
    most_specific_level = min(tag_level(t) for t in repeated)
    specificity = SPECIFICITY_FACTOR[most_specific_level]
    best_ratio = max(
        c / anchor_count
        for tag_id, c in repeated.items()
        if tag_level(tag_id) == most_specific_level
    )
    return specificity * 2.0 * math.log1p(9.0 * best_ratio) / math.log1p(9.0)


def _multi_anchor_themes_repeated(
    themes_trait_sets: list[set[tuple[int, int]]],
) -> set[tuple[int, int]]:
    """Identify ``(kind, trait_id)`` pairs repeated across ≥2 anchors.

    Trait keys are already kind-namespaced by ``_themes_traits_for_movie``,
    so counting collisions across the per-anchor sets gives the right
    answer without any kind disambiguation step.
    """
    counts: dict[tuple[int, int], int] = {}
    for traits in themes_trait_sets:
        for pair in traits:
            counts[pair] = counts.get(pair, 0) + 1
    return {pair for pair, c in counts.items() if c >= 2}


# V3.1 cohesion-IDF tradeoff for multi-anchor themes recall: rarer
# traits qualify with weaker cohesion. The intuition (per design
# discussion): a low-IDF trait is too noisy to fetch on unless ALL
# anchors carry it; a high-IDF trait is distinctive enough that even
# half the cohort sharing it is meaningful.
THEMES_RECALL_COHESION_BAR_LOW  = 1.00  # idf < 0.30: all anchors must carry
THEMES_RECALL_COHESION_BAR_MOD  = 0.67  # 0.30 ≤ idf < 0.55: ≥ 2/3 of anchors
THEMES_RECALL_COHESION_BAR_HIGH = 0.50  # idf ≥ 0.55: ≥ half of anchors


def _multi_anchor_consensus_themes_traits(
    themes_trait_sets: list[set[tuple[int, int]]],
    idfs: dict[tuple[int, int], float],
    n: int,
) -> set[tuple[int, int]]:
    """V3.1 consensus trait pool with cohesion-IDF tradeoff.

    For each (kind, trait_id) appearing in any anchor's themes pool,
    compute cohesion = M_t / N (fraction of anchors carrying the
    trait). Apply an IDF-scaled cohesion bar: rarer traits clear the
    bar at lower cohesion. Returns the consensus set used for
    multi-anchor themes-recall fetching — which then feeds the same
    `fetch_movie_ids_by_themes_recall` SQL aggregate as the
    single-anchor flow.

    The bar tiers use the generic IDF_TIER_* boundaries (low < 0.30 ≤
    moderate < 0.55 ≤ rare) so cohesion semantics align with the
    bucket's notion of "low / moderate / rare" rarity.
    """
    if n <= 0:
        return set()
    counts: dict[tuple[int, int], int] = {}
    for traits in themes_trait_sets:
        for pair in traits:
            counts[pair] = counts.get(pair, 0) + 1
    consensus: set[tuple[int, int]] = set()
    for pair, count in counts.items():
        cohesion = count / n
        idf = idfs.get(pair, 0.0)
        if idf < IDF_TIER_LOW_MAX:
            bar = THEMES_RECALL_COHESION_BAR_LOW
        elif idf < IDF_TIER_MODERATE_MAX:
            bar = THEMES_RECALL_COHESION_BAR_MOD
        else:
            bar = THEMES_RECALL_COHESION_BAR_HIGH
        if cohesion >= bar:
            consensus.add(pair)
    return consensus


def _multi_anchor_themes_scores(
    *,
    candidate_rows: dict[int, dict],
    repeated_traits: set[tuple[int, int]],
    idf_lookup: dict[tuple[int, int], float],
) -> dict[int, float]:
    """V2 themes lane: shared-IDF / total-IDF mass over repeated traits.

    A candidate matching every repeated trait scores 1.0. Common tags
    with idf ≈ 0 contribute negligibly even when matched, so anchor sets
    bound by niche shared traits (FOLK_HORROR, KAIJU, CYBERPUNK) carry
    real signal while anchor sets that share only DRAMA/COMEDY collapse
    to ~0 even at perfect overlap.

    Trait keys are kind-namespaced (``(kind, id)``) so different families
    sharing a numeric ID don't cross-contaminate.
    """
    if not repeated_traits:
        return {}
    denom = sum(idf_lookup.get(pair, 0.0) for pair in repeated_traits)
    if denom <= 0.0:
        return {}
    out: dict[int, float] = {}
    for movie_id, row in candidate_rows.items():
        candidate_traits = _themes_traits_for_movie(row)
        shared = candidate_traits & repeated_traits
        if not shared:
            continue
        numer = sum(idf_lookup.get(pair, 0.0) for pair in shared)
        base = numer / denom
        if base > 0.0:
            # V3.4.5 compounding multiplier on aggregate moderate+ overlap.
            mult = _themes_combo_multiplier(shared, idf_lookup)
            out[movie_id] = _clamp(base * mult)
    return out


def _multi_anchor_specific_award_scores(
    *,
    candidate_award_tags_by_movie: dict[int, set[int]],
    anchor_trait_sets: list[set[int]],
    anchor_count: int,
) -> dict[int, float]:
    """Tier-weighted candidate scoring for the specific-award lane.

    Repeated tags across anchors form the score basis. Each level
    contributes its TIER_WEIGHT (L0=1.00, L1=0.50, L2=0.20) so a
    candidate matching the same L0 tag as the consensus dominates one
    that only shares the L2 group bucket. Unmatched candidates score 0.
    """
    if anchor_count < 2 or not anchor_trait_sets:
        return {}
    counts: dict[int, int] = {}
    for traits in anchor_trait_sets:
        for tag_id in traits:
            counts[tag_id] = counts.get(tag_id, 0) + 1
    repeated = {tag_id for tag_id, c in counts.items() if c >= 2}
    if not repeated:
        return {}
    denom = sum(TIER_WEIGHT[tag_level(t)] for t in repeated)
    if denom <= 0.0:
        return {}
    out: dict[int, float] = {}
    for movie_id, candidate_tags in candidate_award_tags_by_movie.items():
        shared = candidate_tags & repeated
        if not shared:
            continue
        numer = sum(TIER_WEIGHT[tag_level(t)] for t in shared)
        score = numer / denom
        if score > 0.0:
            out[movie_id] = _clamp(score)
    return out


def _per_anchor_active_anchor_types(
    anchor_ids: list[int],
    anchor_rows: dict[int, dict],
    director_terms_by_anchor: dict[int, set[int]],
    anchor_studio_entries: dict[int, list[_ResolvedStudioEntry]],
) -> dict[int, list[AnchorType]]:
    """Build the per-anchor active-types breakdown for the debug payload.

    Each anchor's individual flags surface on the multi-anchor result so
    failures like "Schindler's List + 12 Years a Slave dominated the
    centroid" are diagnosable from the response without having to rerun
    each anchor as a single-anchor query.
    """
    out: dict[int, list[AnchorType]] = {}
    for anchor_id in anchor_ids:
        row = anchor_rows[anchor_id]
        types: list[AnchorType] = ["standard_shape"]
        bucket = _quality_bucket(row)
        if bucket in {"cult_garbage", "prestige"}:
            types.append(bucket)  # type: ignore[arg-type]
        a_lin, a_uni, a_sub = _franchise_traits(row)
        if a_lin or a_uni or a_sub:
            types.append("franchise_dominant")
        if anchor_studio_entries.get(anchor_id):
            types.append("studio_lineage")
        if _as_int_set(row.get("source_material_type_ids")):
            types.append("source_material")
        out[anchor_id] = types
    return out


async def _low_cohesion_fallback(
    anchor_ids: list[int],
    anchor_rows: dict[int, dict],
    vectors_by_anchor: dict[int, dict[VectorName, list[float]]],
    studio_entries_by_company_id: dict[int, list[StudioSimilarityEntry]],
    director_terms_by_anchor: dict[int, set[int]],
    *,
    limit: int,
    qdrant_limit: int,
    quality_limit: int,
    mean_pairwise_cosine: float,
    metadata_filters: MetadataFilters | None = None,
) -> SimilarMoviesSearchResult:
    """Fallback for chaotic anchor sets where the centroid lands in noise.

    Runs each anchor as an independent single-anchor query, then weaves
    the per-anchor lists round-robin by rank. Returns the result with a
    debug flag set so callers know the fallback fired; UI presents
    results identically to a normal multi-anchor return.
    """
    n = len(anchor_ids)
    per_anchor_limit = max(1, math.ceil(limit * 1.2 / n))
    per_anchor_results = await asyncio.gather(
        *(
            _run_single_anchor_similarity(
                anchor_id,
                anchor_rows,
                vectors_by_anchor,
                studio_entries_by_company_id,
                director_terms_by_anchor,
                limit=per_anchor_limit,
                qdrant_limit=qdrant_limit,
                quality_limit=quality_limit,
                metadata_filters=metadata_filters,
            )
            for anchor_id in anchor_ids
        )
    )

    # Round-robin interleave: take the i-th result from each anchor in
    # turn, dedupe by movie_id, stop once we hit the requested limit.
    seen: set[int] = set(anchor_ids)
    interleaved: list[SimilarMovieResult] = []
    for rank in range(per_anchor_limit):
        for result in per_anchor_results:
            if rank >= len(result.ranked):
                continue
            entry = result.ranked[rank]
            if entry.movie_id in seen:
                continue
            seen.add(entry.movie_id)
            interleaved.append(entry)
            if len(interleaved) >= limit:
                break
        if len(interleaved) >= limit:
            break

    return SimilarMoviesSearchResult(
        anchor_movie_ids=anchor_ids,
        ranked=interleaved,
        active_anchor_types=["standard_shape"],
        debug=SimilarMoviesDebug(
            vector_space_weights={},
            vector_space_cohesion={"mean_pairwise_cosine": mean_pairwise_cosine},
            low_cohesion_fallback_used=True,
        ),
    )


# ---------------------------------------------------------------------------
# V3.4 Bucket-Weaver helpers
# ---------------------------------------------------------------------------


def _rare_tier_idf_sum(
    traits: set[tuple[int, int]],
    idf_lookup: dict[tuple[int, int], float],
) -> float:
    """Sum IDFs for traits in the truly-rare tier
    (idf >= WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN, i.e., 0.55).

    Drives the rare_keyword bucket *signal* — aligned with the bucket
    *membership* threshold so "anchor has rare traits → bucket fires"
    and "candidate shares a rare trait → bucket eligible" reference
    the same set of traits. Pre-V3.4.4 the signal used a moderate+high
    threshold (0.30) while membership used 0.55, which silently no-op'd
    the bucket on tag-rich-but-no-truly-rare-tag anchors.
    """
    return sum(
        idf for trait in traits
        for idf in [idf_lookup.get(trait, 0.0)]
        if idf >= WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN
    )


def _candidate_franchise_score_against_anchors(
    candidate_row: dict,
    anchor_rows: list[dict],
) -> float:
    """Max V3 structural franchise score across all anchors.

    For multi-anchor cohorts, a candidate's franchise membership is
    determined by its strongest pairwise structural match — a Star
    Wars film paired with a Pixar anchor and a Star Wars anchor scores
    1.00 against the SW anchor and is therefore franchise-bucket
    eligible at score >= 0.55.
    """
    best = 0.0
    for anchor_row in anchor_rows:
        score = _franchise_score_v2(anchor_row, candidate_row)
        if score > best:
            best = score
    return best


def _compute_single_anchor_bucket_data(
    *,
    anchor_row: dict,
    anchor_directors: set[int],
    auteur_term_ids: frozenset[int],
    anchor_themes_traits: set[tuple[int, int]],
    themes_idfs: dict[tuple[int, int], float],
    candidate_rows: dict[int, dict],
    director_candidate_terms: dict[int, set[int]],
    franchise_candidate_ids: set[int],
) -> tuple[dict[str, float], dict[int, set[str]]]:
    """V3.4: build bucket signal scores + per-candidate memberships for
    single-anchor flow.

    Signals (each in [0, 1]):
    - auteur: 1.0 iff anchor's director is curated, else 0.0.
    - franchise: clip(catalog_size/5, 0, 1) × max_member_pop_percentile,
      where catalog_size and max_member are computed over candidates
      with `_franchise_score_v2 >= 0.55` against the anchor.
    - rare_keyword: anchor's rare-tier IDF sum / RARE_KW_BUCKET_SIGNAL_SCALE,
      where rare-tier means idf >= WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN
      (aligned with the bucket's membership threshold in V3.4.4).
    - lead_actor: 0.0 (multi-only).
    """
    signals: dict[str, float] = {b: 0.0 for b in ALL_BUCKETS}
    memberships: dict[int, set[str]] = {
        mid: set() for mid in candidate_rows
    }

    # Auteur signal + memberships.
    auteur_intersect = anchor_directors & auteur_term_ids
    if auteur_intersect:
        signals[BUCKET_AUTEUR] = 1.0
        for mid, dirs in director_candidate_terms.items():
            if dirs & auteur_intersect:
                memberships.setdefault(mid, set()).add(BUCKET_AUTEUR)

    # Franchise signal + memberships.
    if franchise_candidate_ids:
        franchise_pool: list[float] = []
        for mid in franchise_candidate_ids:
            row = candidate_rows.get(mid)
            if row is None:
                continue
            score = _franchise_score_v2(anchor_row, row)
            if score >= WEAVER_FRANCHISE_BUCKET_MIN_SCORE:
                memberships.setdefault(mid, set()).add(BUCKET_FRANCHISE)
                franchise_pool.append(row.get("popularity_percentile") or 0.0)
        if franchise_pool:
            catalog_size_factor = min(1.0, len(franchise_pool) / 5.0)
            max_pop = max(franchise_pool)
            signals[BUCKET_FRANCHISE] = catalog_size_factor * max_pop

    # Rare-keyword signal + memberships. V3.4.4: signal and membership
    # both filter on the same idf >= WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN
    # threshold so the bucket is consistent — fires only when the
    # anchor carries genuinely-rare traits, and eligible candidates
    # are exactly those that share one of them.
    rare_idf_sum = _rare_tier_idf_sum(anchor_themes_traits, themes_idfs)
    if rare_idf_sum > 0.0:
        signals[BUCKET_RARE_KEYWORD] = min(
            1.0, rare_idf_sum / RARE_KW_BUCKET_SIGNAL_SCALE
        )
        anchor_rare_traits = {
            t for t in anchor_themes_traits
            if themes_idfs.get(t, 0.0) >= WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN
        }
        if anchor_rare_traits:
            for mid, row in candidate_rows.items():
                if anchor_rare_traits & _themes_traits_for_movie(row):
                    memberships.setdefault(mid, set()).add(BUCKET_RARE_KEYWORD)

    # Best-overall is universal — handled inside the weaver.
    return signals, memberships


def _compute_multi_anchor_bucket_data(
    *,
    anchor_ids: list[int],
    anchor_rows: dict[int, dict],
    director_trait_sets: list[set[int]],
    auteur_term_ids: frozenset[int],
    themes_trait_sets: list[set[tuple[int, int]]],
    themes_idfs: dict[tuple[int, int], float],
    candidate_rows: dict[int, dict],
    director_candidate_terms: dict[int, set[int]],
    franchise_candidate_ids: set[int],
    candidate_top_billing: dict[int, set[int]],
    cast_trait_sets: list[set[int]],
) -> tuple[dict[str, float], dict[int, set[str]]]:
    """V3.4: build bucket signals + per-candidate memberships for
    multi-anchor cohorts.

    Multi-anchor signal scaling:
    - auteur: M_d_auteur / N where M_d_auteur is the count of anchors
      with at least one curated director.
    - franchise: clip(catalog_size/5, 0, 1) × max_member_pop × M_f/N
      where M_f is anchors carrying any lineage_entry_id.
    - rare_keyword: V3.4.2 cohesion-weighted IDF sum filtered to
      idf >= WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN (V3.4.4 alignment).
    - lead_actor: M_a/N where M_a is the max count of anchors sharing
      any single top-3-billed actor.
    """
    n = max(1, len(anchor_ids))
    signals: dict[str, float] = {b: 0.0 for b in ALL_BUCKETS}
    memberships: dict[int, set[str]] = {
        mid: set() for mid in candidate_rows
    }

    # Auteur — count anchors with any auteur director, weighted by N.
    auteur_anchor_count = sum(
        1 for dirs in director_trait_sets if dirs & auteur_term_ids
    )
    auteur_union = set().union(*director_trait_sets) & auteur_term_ids
    if auteur_anchor_count > 0 and auteur_union:
        signals[BUCKET_AUTEUR] = auteur_anchor_count / n
        for mid, dirs in director_candidate_terms.items():
            if dirs & auteur_union:
                memberships.setdefault(mid, set()).add(BUCKET_AUTEUR)

    # Franchise — anchor cohesion + catalog size + popularity.
    franchise_anchor_count = sum(
        1 for aid in anchor_ids
        if _as_int_set(anchor_rows[aid].get("lineage_entry_ids"))
    )
    if franchise_anchor_count > 0 and franchise_candidate_ids:
        anchor_row_list = [anchor_rows[aid] for aid in anchor_ids]
        franchise_pool: list[float] = []
        for mid in franchise_candidate_ids:
            row = candidate_rows.get(mid)
            if row is None:
                continue
            score = _candidate_franchise_score_against_anchors(
                row, anchor_row_list
            )
            if score >= WEAVER_FRANCHISE_BUCKET_MIN_SCORE:
                memberships.setdefault(mid, set()).add(BUCKET_FRANCHISE)
                franchise_pool.append(row.get("popularity_percentile") or 0.0)
        if franchise_pool:
            catalog_size_factor = min(1.0, len(franchise_pool) / 5.0)
            max_pop = max(franchise_pool)
            cohesion = franchise_anchor_count / n
            signals[BUCKET_FRANCHISE] = catalog_size_factor * max_pop * cohesion

    # Rare-keyword — V3.4.2 cohesion-weighted IDF sum, V3.4.4 aligned
    # IDF threshold. A trait that appears in only 1/N anchors gets
    # weight 0; weight scales linearly to 1.0 at full N/N cohesion via
    # cohesion_weight = max(0, (M_t - 1) / (N - 1)). This prevents
    # one-anchor outlier traits (Everyone Says I Love You's musical/song
    # traits in the Slasher cohort) from firing the bucket on
    # heterogeneous cohorts. V3.4.4: filter on idf >=
    # WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN (0.55) so signal and membership
    # reference the same trait set — pre-V3.4.4 the signal admitted
    # idf >= 0.30 (moderate-tier) traits that membership then excluded,
    # silently no-op'ing the bucket on tag-rich anchors.
    trait_anchor_counts: dict[tuple[int, int], int] = {}
    for traits in themes_trait_sets:
        for t in traits:
            trait_anchor_counts[t] = trait_anchor_counts.get(t, 0) + 1
    cohesion_denom = max(1, n - 1)
    weighted_idf_sum = 0.0
    for t, m_t in trait_anchor_counts.items():
        idf = themes_idfs.get(t, 0.0)
        if idf < WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN:
            continue
        cohesion_weight = max(0.0, (m_t - 1) / cohesion_denom)
        if cohesion_weight == 0.0:
            continue
        weighted_idf_sum += idf * cohesion_weight
    if weighted_idf_sum > 0.0:
        signals[BUCKET_RARE_KEYWORD] = min(
            1.0, weighted_idf_sum / RARE_KW_BUCKET_SIGNAL_SCALE
        )
        # Membership: candidate must share a rare-tier trait that at
        # least 2 anchors carry. Same M_t >= 2 floor as the signal.
        anchor_rare_traits = {
            t for t, m_t in trait_anchor_counts.items()
            if m_t >= 2
            and themes_idfs.get(t, 0.0) >= WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN
        }
        if anchor_rare_traits:
            for mid, row in candidate_rows.items():
                if anchor_rare_traits & _themes_traits_for_movie(row):
                    memberships.setdefault(mid, set()).add(BUCKET_RARE_KEYWORD)

    # Lead actor — multi-only. Reuses the cast lane's M_a / shared_leads
    # logic. Bucket fires when cohesion >= CAST_FLOOR_RATIO_THRESHOLD
    # (0.5), matching the cast floor's gate; membership is candidates
    # sharing any lead the cohort has cohesion ≥0.5 in.
    if cast_trait_sets:
        m_per_actor: dict[int, int] = {}
        for billing in cast_trait_sets:
            for actor in billing:
                m_per_actor[actor] = m_per_actor.get(actor, 0) + 1
        if m_per_actor:
            max_m = max(m_per_actor.values())
            ratio = max_m / n
            if ratio >= CAST_FLOOR_RATIO_THRESHOLD:
                signals[BUCKET_LEAD_ACTOR] = ratio
                shared_leads = {
                    a for a, m in m_per_actor.items() if m == max_m
                }
                for mid, billing in candidate_top_billing.items():
                    if billing & shared_leads:
                        memberships.setdefault(mid, set()).add(BUCKET_LEAD_ACTOR)

    return signals, memberships


async def _empty_set() -> set[int]:
    return set()


async def _empty_dict() -> dict[int, set[int]]:
    return {}


async def _empty_movie_set_dict() -> dict[int, set[int]]:
    return {}


async def _empty_award_signals() -> dict[int, SimilarityAwardSignals]:
    return {}


async def _empty_idf_dict() -> dict[tuple[int, int], float]:
    return {}
