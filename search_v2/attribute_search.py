# Search V2 — Attribute search flow.
#
# Backs the `POST /attribute_search` endpoint: a deterministic browse
# path that takes a set of hard attributes (genres, audio languages,
# streaming services, release/runtime/maturity ranges, plus named
# people with optional role restriction) and returns the top movies
# ranked by the 80/20 popularity-vs-reception neutral prior.
#
# No NLP, no LLM, no vector search. Every input is either a closed
# enum / numeric range (handled by MetadataFilters at the SQL layer)
# or a person name resolved against the lexical posting tables.
#
# Flow:
#
#   1. Normalize + dedupe person inputs. `(normalized_name, role)`
#      tuples are deduped so callers that pass the same person twice
#      don't double-query Postgres.
#
#   2. Batched name → term_id resolution. One `fetch_phrase_term_ids`
#      call covers every distinct normalized name across all persons.
#
#   3. Per-person movie_id set lookup.
#        - role set    → query that one role's posting table.
#        - role unset  → fan out across all five role posting tables
#          in parallel and UNION the resulting sets (any credit
#          qualifies). Set-only — no prominence scoring, unlike the
#          person endpoint's bucket model.
#      `metadata_filters` is applied inline inside each posting
#      lookup via `_build_inline_movie_card_filter_clause`, so the
#      per-person set is already filter-respecting.
#
#   4. Intersection across persons (AND). A movie must satisfy every
#      person filter the caller supplied.
#
#   5. Rank via the neutral 80/20 prior. When no person filters fire,
#      we skip the restrict set entirely and let
#      `fetch_neutral_reranker_seed_ids` rank the whole movie_card
#      table (filter-respecting). When the intersection is non-empty
#      we pass it as `restrict_movie_ids` to scope the same ranking
#      to just that pool.
#
# Wire-layer schemas (`PersonInput`, `AttributeSearchBody`) live in
# api/main.py — this module only sees the post-validation
# `PersonSpec` dataclass below so we stay decoupled from Pydantic /
# FastAPI internals.

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from db.postgres import (
    PEOPLE_POSTING_TABLES,
    PostingTable,
    fetch_movie_ids_by_term_ids,
    fetch_neutral_reranker_seed_ids,
    fetch_phrase_term_ids,
)
from implementation.classes.schemas import MetadataFilters
from implementation.misc.helpers import normalize_string
from schemas.entity_translation import PersonCategory


# Default cap on results returned to the API. Browse-style endpoint —
# 250 is enough for the UI to render a paginated grid without a
# follow-up call, and the underlying ranking query handles the LIMIT
# in SQL so there is no client-side trim cost.
DEFAULT_ATTRIBUTE_SEARCH_LIMIT = 250


# Specific-role → posting table dispatch. Unlike the entity endpoint's
# `_PERSON_CATEGORY_TO_TABLE` (which excludes ACTOR because that path
# computes billing-band prominence), attribute search treats every
# role as a flat set-membership lookup, so ACTOR is included here.
# UNKNOWN is handled separately — see `_resolve_person_movie_ids`.
_PERSON_CATEGORY_TO_TABLE: dict[PersonCategory, PostingTable] = {
    PersonCategory.ACTOR: PostingTable.ACTOR,
    PersonCategory.DIRECTOR: PostingTable.DIRECTOR,
    PersonCategory.WRITER: PostingTable.WRITER,
    PersonCategory.PRODUCER: PostingTable.PRODUCER,
    PersonCategory.COMPOSER: PostingTable.COMPOSER,
}


@dataclass(frozen=True, slots=True)
class PersonSpec:
    """One named-person filter, post-validation.

    `name` is the raw name as supplied by the caller (already
    whitespace-stripped at the API boundary; normalization happens
    inside `run_attribute_search`).

    `role` is the resolved PersonCategory: a specific role restricts
    the lookup to that posting table; UNKNOWN unions across all five
    role tables (any credit qualifies).
    """
    name: str
    role: PersonCategory


async def _resolve_person_movie_ids(
    spec: PersonSpec,
    term_ids: list[int],
    *,
    metadata_filters: Optional[MetadataFilters],
) -> set[int]:
    """Resolve one PersonSpec's term_ids into the movie_id set.

    Specific role → one posting table lookup. UNKNOWN → parallel
    lookup across every PEOPLE_POSTING_TABLES entry, unioned.

    `metadata_filters` flows into each `fetch_movie_ids_by_term_ids`
    call so the per-person set is already filter-respecting — the
    final intersection / ranking never re-filters.
    """
    if not term_ids:
        return set()

    if spec.role == PersonCategory.UNKNOWN:
        # Fan out across all five role tables in parallel. Any credit
        # for this person on a movie qualifies — actor, director,
        # writer, producer, or composer. No prominence weighting; we
        # only care about set membership.
        per_table = await asyncio.gather(
            *(
                fetch_movie_ids_by_term_ids(
                    table, term_ids, metadata_filters=metadata_filters,
                )
                for table in PEOPLE_POSTING_TABLES
            )
        )
        union: set[int] = set()
        for table_set in per_table:
            union |= table_set
        return union

    # Specific role → single posting table. KeyError here would
    # mean a new PersonCategory value was introduced without
    # updating the dispatch dict; surface it loudly rather than
    # silently returning the empty set.
    table = _PERSON_CATEGORY_TO_TABLE[spec.role]
    return await fetch_movie_ids_by_term_ids(
        table, term_ids, metadata_filters=metadata_filters,
    )


async def run_attribute_search(
    *,
    people: list[PersonSpec],
    metadata_filters: Optional[MetadataFilters],
    limit: int = DEFAULT_ATTRIBUTE_SEARCH_LIMIT,
) -> list[int]:
    """Orchestrate the attribute-search flow → ranked movie_ids.

    Args:
      people: Zero or more `PersonSpec` filters. Multiple persons
        intersect (AND): the returned movies must satisfy every
        person filter.
      metadata_filters: Optional hard filters (genres, languages,
        streaming services, release_date / runtime / maturity
        ranges). Applied at the SQL layer inside every posting
        lookup AND inside the final ranking query.
      limit: Cap on returned movie_ids (default 250).

    Returns:
      List of `movie_id`s ranked by the 80/20 popularity/reception
      neutral prior (descending). Empty list when the people
      intersection is empty, when a supplied name fails to resolve,
      or when filters exclude every candidate.
    """
    # No-person fast path: skip resolution / intersection entirely
    # and let the ranking query scan the whole movie_card table
    # (filter-respecting). This is the "browse everything" case —
    # equivalent to running the V2 neutral fallback path directly.
    if not people:
        return await fetch_neutral_reranker_seed_ids(
            limit=limit, metadata_filters=metadata_filters,
        )

    # Normalize + dedupe (normalized_name, role) tuples. Same person
    # passed twice with the same role should issue one set of
    # lookups, not two. Different roles for the same name remain
    # distinct entries since they resolve through different posting
    # tables.
    deduped: list[PersonSpec] = []
    seen: set[tuple[str, PersonCategory]] = set()
    for spec in people:
        normalized = normalize_string(spec.name)
        if not normalized:
            # Empty after normalization — same effect as an
            # unresolvable name. Inject a sentinel so the
            # intersection collapses to empty per the user's
            # "treat as zero matches" contract.
            return []
        key = (normalized, spec.role)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(PersonSpec(name=normalized, role=spec.role))

    # Batch name → term_id resolution. One Postgres round trip
    # covers every distinct normalized name across all persons,
    # even when they appear under different roles.
    distinct_names = list({spec.name for spec in deduped})
    name_to_term_id = await fetch_phrase_term_ids(distinct_names)

    # Per-person movie_id set lookup, in parallel. Each task pulls
    # the term_id for its name from the shared resolution map; a
    # missing name resolves to an empty term_id list and short-
    # circuits inside `_resolve_person_movie_ids` to an empty set.
    def _term_ids_for(spec: PersonSpec) -> list[int]:
        # fetch_phrase_term_ids returns one term_id per known name
        # and omits unresolvable names. Wrap a hit in a list to match
        # the posting-table helper's signature.
        term_id = name_to_term_id.get(spec.name)
        return [term_id] if term_id is not None else []

    per_person = await asyncio.gather(
        *(
            _resolve_person_movie_ids(
                spec,
                _term_ids_for(spec),
                metadata_filters=metadata_filters,
            )
            for spec in deduped
        )
    )

    # AND across persons. Any person with an empty match set
    # collapses the intersection to empty — including unresolvable
    # names (their per-person set is empty). Iterate from the
    # smallest set to short-circuit cheaply when the intersection
    # empties out partway through. We copy the smallest set into a
    # fresh `intersection` instead of aliasing per_person[0] — keeps
    # the loop body's `&=` from mutating an element of the input
    # list, which would surprise any future code that touches
    # per_person after this loop.
    per_person.sort(key=len)
    intersection: set[int] = set(per_person[0])
    for movie_ids in per_person[1:]:
        intersection &= movie_ids
        if not intersection:
            break

    if not intersection:
        return []

    # Rank the intersection by the neutral 80/20 prior. metadata
    # filters are already baked into the per-person sets so this
    # call only needs the restrict + limit; passing filters again
    # is redundant but harmless (the SQL re-applies them, matching
    # the same rows the intersection already contains).
    return await fetch_neutral_reranker_seed_ids(
        limit=limit,
        restrict_movie_ids=intersection,
        metadata_filters=metadata_filters,
    )
