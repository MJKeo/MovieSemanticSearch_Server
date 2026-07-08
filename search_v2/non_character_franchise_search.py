# Search V2 — Non-character franchise search flow.
#
# Owns the non_character_franchise path described by Step 0's
# NonCharacterFranchiseFlowData. The flow has two stages:
#
#   1. Alias expansion (LLM). The Step 0 canonical_name is fed to the
#      existing Step 3 franchise translator
#      (search_v2/endpoint_fetching/franchise_query_generation.py) so it
#      emits 1-3 canonical alternate forms in `franchise_names` — e.g.
#      "MCU" → ["marvel cinematic universe", "marvel"]. This closes the
#      gap where Step 0's canonical_name and the ingest-side
#      `normalized_string` use different surface forms; the shared
#      tokenizer alone can't bridge abbreviation-style variants.
#
#      Only the `franchise_names` axis is consumed. Every other axis on
#      the FranchiseQuerySpec (subgroup_names, lineage_position,
#      structural_flags, launch_scope, prefer_lineage) is deliberately
#      ignored — Step 0 already committed that this is a non-character
#      franchise name reference, nothing else.
#
#      LLM failure (timeout, parse error, etc.) is soft: the executor
#      falls back to `[canonical_name]` so the deterministic single-name
#      path still works. We never block on the alias-expansion step.
#
#   2. Bucketed lookup (Postgres). The (deduped, normalized) name list
#      is OR-unioned against lex.franchise_entry.normalized_string and
#      joined into public.movie_card via the GIN-indexed entry-id
#      arrays. Returns two ordered buckets:
#
#        primary_franchise   = movies whose lineage_entry_ids contain
#                              any matched franchise_entry_id (sorted
#                              by popularity DESC).
#        secondary_franchise = movies whose shared_universe_entry_ids
#                              contain any matched entry but whose
#                              lineage does NOT (the extended universe).
#
# Append-after-sort: callers consume primary_franchise + secondary_franchise
# in that exact order. The least-popular primary movie still precedes the
# most-popular secondary movie. Score per movie is intentionally NOT
# emitted — bucket-then-popularity ordering carries the relevance signal.
#
# Latency budget: ~1-2s for the LLM call plus one Postgres round-trip
# (see db.postgres.fetch_non_character_franchise_movies). The two GIN
# indexes on movie_card.lineage_entry_ids and
# movie_card.shared_universe_entry_ids do the SQL heavy lifting;
# lex.franchise_entry.normalized_string is UNIQUE so each name probe is
# a single index lookup.

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from opentelemetry import trace

from db.postgres import fetch_non_character_franchise_movies
from implementation.classes.schemas import MetadataFilters
from implementation.misc.franchise_text import normalize_franchise_string
from observability.names import (
    QUERY_SEARCH_BRANCH_ALIASES,
    QUERY_SEARCH_BRANCH_ENTITIES,
    QUERY_SEARCH_BRANCH_ENTITY_RESOLVED_COUNTS,
    QUERY_SEARCH_BRANCH_FRANCHISE_LLM_FALLBACK,
    QUERY_SEARCH_BRANCH_SECONDARY_COUNT,
    QUERY_SEARCH_BRANCH_TOP_TIER,
    QUERY_SEARCH_BRANCH_TOP_TIER_COUNT,
    QUERY_SEARCH_BRANCH_UNRESOLVED_ENTITY_COUNT,
)
from schemas.step_0_flow_routing import NonCharacterFranchiseFlowData
# Reuse the Step-3 category handler's LLM config so the two callers
# of generate_franchise_query stay in lockstep on model + reasoning
# settings — if the handler upgrades models, this caller follows.
from search_v2.endpoint_fetching.category_handlers.handler import (
    HANDLER_LLM_KWARGS,
    HANDLER_LLM_MODEL,
    HANDLER_LLM_PROVIDER,
)
from search_v2.endpoint_fetching.franchise_query_generation import (
    generate_franchise_query,
)

logger = logging.getLogger(__name__)


_BUCKET_PRIMARY = 0
_BUCKET_SECONDARY = 1


@dataclass
class NonCharacterFranchiseSearchResult:
    """Output of run_non_character_franchise_search.

    primary_franchise: movie_ids whose lineage contains any matched
        franchise entry, sorted by popularity_score DESC (NULLS LAST,
        movie_id DESC as tiebreaker).
    secondary_franchise: movie_ids whose shared_universe contains a
        matched entry but whose lineage does not. Disjoint from
        primary_franchise by construction (SQL bool_or assigns a row to
        bucket 0 when any matched entry hits the lineage side — universe
        rows are universe-only).
    """

    primary_franchise: list[int] = field(default_factory=list)
    secondary_franchise: list[int] = field(default_factory=list)


async def run_non_character_franchise_search(
    flow_data: NonCharacterFranchiseFlowData,
    *,
    limit: int = 100,
    metadata_filters: MetadataFilters | None = None,
) -> NonCharacterFranchiseSearchResult:
    """Execute the non-character franchise flow.

    Calls the Step 3 franchise translator to expand the single
    canonical_name into 1-3 canonical alternate forms, normalizes each,
    and OR-unions them against lex.franchise_entry. Falls back to the
    deterministic [canonical_name] path on any LLM failure.

    Args:
        flow_data: From Step0Response.to_non_character_franchise_flow_data().
            Carries the canonical_name the LLM resolved (e.g. "Marvel",
            "MCU", "Mission: Impossible").
        limit: Maximum total rows across both buckets. Default 100.

    Returns:
        NonCharacterFranchiseSearchResult with the two ordered buckets.
        Both buckets are empty when no expanded form resolves to a known
        franchise_entry.
    """
    canonical_name = flow_data.canonical_name

    # Stage 1: expand canonical_name into alternate canonical forms via
    # the existing Step 3 franchise translator. The translator expects
    # an intent_rewrite + description + route_rationale; we synthesize
    # all three from the single canonical_name we have. The LLM's job
    # here is narrow — emit franchise_names — so the synthesized
    # context stays minimal.
    expanded_names, llm_fallback = await _expand_canonical_names(canonical_name)
    _record_nc_franchise_identity(canonical_name, expanded_names, llm_fallback)

    # Stage 2: normalize each expanded form. Empty / blank normalizations
    # are dropped by the DB helper. Use a list (not a set) so insertion
    # order survives logs; the helper dedupes internally.
    normalized = [normalize_franchise_string(name) for name in expanded_names]
    normalized = [n for n in normalized if n]
    if not normalized:
        return NonCharacterFranchiseSearchResult()

    rows = await fetch_non_character_franchise_movies(
        normalized, limit=limit, metadata_filters=metadata_filters,
    )

    # Rows are already sorted (bucket asc, popularity desc, movie_id desc).
    # Split into the two buckets in one pass while preserving order.
    primary: list[int] = []
    secondary: list[int] = []
    for movie_id, bucket in rows:
        if bucket == _BUCKET_PRIMARY:
            primary.append(movie_id)
        else:
            secondary.append(movie_id)

    _record_nc_franchise_buckets(primary, secondary)
    return NonCharacterFranchiseSearchResult(
        primary_franchise=primary,
        secondary_franchise=secondary,
    )


def _record_nc_franchise_identity(
    canonical_name: str, expanded_names: list[str], llm_fallback: bool
) -> None:
    """Set the entity identity + alias expansion on the branch span. The single
    canonical name is the identity; the expanded forms are the aliases;
    `llm_fallback` marks that no useful LLM aliases were obtained (the flow ran
    on the bare canonical name), which — for a name like "MCU" that needs the
    abbreviation bridged — typically means empty/thin results."""
    span = trace.get_current_span()
    span.set_attribute(QUERY_SEARCH_BRANCH_ENTITIES, [canonical_name])
    span.set_attribute(QUERY_SEARCH_BRANCH_ALIASES, list(expanded_names))
    span.set_attribute(QUERY_SEARCH_BRANCH_FRANCHISE_LLM_FALLBACK, llm_fallback)


def _record_nc_franchise_buckets(
    primary: list[int], secondary: list[int]
) -> None:
    """Set the bucket sizes on the branch span: primary (lineage) is the top
    tier; secondary is the universe-only extension. `primary` empty with
    `secondary` populated means the name matched a universe tag but nothing
    carries it as core lineage — a tagging/coverage gap. `resolved_counts`
    carries the total matched-movie count for the (single) entity."""
    span = trace.get_current_span()
    total = len(primary) + len(secondary)
    span.set_attribute(QUERY_SEARCH_BRANCH_ENTITY_RESOLVED_COUNTS, [total])
    span.set_attribute(
        QUERY_SEARCH_BRANCH_UNRESOLVED_ENTITY_COUNT, 0 if total else 1
    )
    span.set_attribute(QUERY_SEARCH_BRANCH_TOP_TIER, "primary")
    span.set_attribute(QUERY_SEARCH_BRANCH_TOP_TIER_COUNT, len(primary))
    span.set_attribute(QUERY_SEARCH_BRANCH_SECONDARY_COUNT, len(secondary))


async def _expand_canonical_names(canonical_name: str) -> tuple[list[str], bool]:
    """Run the Step 3 franchise translator to get alt canonical forms.

    Returns ``(names, llm_fallback)``. ``names`` is 1-3 canonical surface forms
    drawn from the LLM's `franchise_names` axis. On any failure (timeout, parse
    error, the LLM returning a null axis), falls back to `[canonical_name]` so
    the deterministic non-LLM path still works, and ``llm_fallback`` is True.
    The deterministic fallback is essential: the rest of the pipeline assumes
    this function always returns at least the user's original canonical_name.
    ``llm_fallback`` covers both the exception path and the null-axis path —
    in both, no useful LLM aliases were obtained (the abbreviation bridge that
    makes "MCU" resolve to "marvel cinematic universe" is lost).

    Synthesized inputs to generate_franchise_query:
      - intent_rewrite: positive-presence framing of the franchise lookup
      - description: positive-presence statement of the requirement
      - route_rationale: a hint that the upstream routing was franchise-
        name-driven (the translator treats this as a soft hint, not
        authority)
    """
    intent_rewrite = f"Movies in the {canonical_name} franchise."
    description = f"is a {canonical_name} movie"
    route_rationale = "franchise name reference"

    try:
        spec, _, _ = await generate_franchise_query(
            intent_rewrite=intent_rewrite,
            description=description,
            route_rationale=route_rationale,
            provider=HANDLER_LLM_PROVIDER,
            model=HANDLER_LLM_MODEL,
            **HANDLER_LLM_KWARGS,
        )
    except Exception:  # noqa: BLE001 — soft-fail by design
        logger.warning(
            "Franchise alias-expansion LLM call failed; falling back to "
            "deterministic single-name path for canonical_name=%r",
            canonical_name,
            exc_info=True,
        )
        return [canonical_name], True

    names = spec.franchise_names
    if not names:
        # Translator can legitimately leave franchise_names null (e.g.
        # when it decides the request is structural-only). For our
        # alias-expansion use case that's effectively a no-op — fall
        # back to the original name.
        return [canonical_name], True
    return list(names), False
