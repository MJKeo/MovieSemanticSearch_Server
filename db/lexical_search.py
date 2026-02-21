import asyncio
from typing import Optional
from implementation.misc.helpers import normalize_string, tokenize_title_phrase
from implementation.misc.sql_like import escape_like
from implementation.classes.enums import EntityCategory
from implementation.classes.schemas import (
    ExtractedEntitiesResponse,
    LexicalCandidate,
    MetadataFilters,
)
from db.postgres import (
    fetch_phrase_term_ids,
    fetch_title_token_ids,
    fetch_title_token_ids_exact,
    fetch_character_match_counts,
    fetch_character_match_counts_by_query,
    fetch_character_term_ids,
    fetch_title_token_match_scores,
    fetch_phrase_postings_match_counts,
    fetch_movie_ids_by_term_ids,
    PostingTable,
)

# ─── Operational constants (from lexical search guide Section 11) ─────────────
MAX_DF = 10_000
TITLE_SCORE_BETA = 2.0
TITLE_SCORE_THRESHOLD = 0.15
TITLE_MAX_CANDIDATES = 10_000
# ===============================
#     PRIVATE HELPER METHODS
# ===============================


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Deduplicate strings while preserving first-seen order."""
    return list(dict.fromkeys(values))


async def _resolve_all_title_tokens(
    include_title_searches: list[list[str]],
) -> list[dict[int, list[int]]]:
    """
    Resolve all title tokens across all title searches concurrently.

    Deduplicates tokens so each unique token is resolved once, then
    distributes results back into per-title-search maps.

    Args:
        include_title_searches: List of token lists (each is one title search).

    Returns:
        List of token_map dicts, one per title search.
    """
    # Collect (search_idx, token_idx, token) for every token
    triples: list[tuple[int, int, str]] = []
    for search_idx, title_tokens in enumerate(include_title_searches):
        for token_idx, token in enumerate(title_tokens):
            triples.append((search_idx, token_idx, token))

    if not triples:
        return []

    # Deduplicate tokens while preserving first occurrence for stable ordering
    unique_tokens = list(dict.fromkeys(t[2] for t in triples))

    # Resolve all unique tokens in one DB round-trip.
    by_query_idx = await fetch_title_token_ids(tokens=unique_tokens, max_df=MAX_DF)
    token_to_ids = {
        unique_tokens[idx]: ids
        for idx, ids in by_query_idx.items()
        if ids
    }

    # Build per-search token maps from triples
    title_token_maps: list[dict[int, list[int]]] = []
    for search_idx, title_tokens in enumerate(include_title_searches):
        token_map: dict[int, list[int]] = {}
        for token_idx, token in enumerate(title_tokens):
            ids = token_to_ids.get(token)
            if ids:
                token_map[token_idx] = ids
        title_token_maps.append(token_map)

    return title_token_maps


async def _resolve_character_term_ids(
    character_phrases: list[str],
) -> dict[int, list[int]]:
    """
    Resolve normalized character query phrases to term_ids via substring
    matching against lex.character_strings.

    Batches all phrases into a single round-trip.  Each phrase is matched
    with LIKE '%phrase%' accelerated by the trigram GIN index (for
    phrases >= 3 chars; shorter ones fall back to a seqscan of the
    small character_strings table).

    Results are capped at CHARACTER_RESOLVE_LIMIT_PER_PHRASE per phrase.

    Args:
        character_phrases: Normalized character phrase strings.

    Returns:
        Dict of {query_phrase_idx: [term_id, ...]} where the index
        corresponds to the position in the input list.
    """
    if not character_phrases:
        return {}

    query_idxs: list[int] = []
    like_patterns: list[str] = []
    for idx, phrase in enumerate(character_phrases):
        query_idxs.append(idx)
        like_patterns.append(f"%{escape_like(phrase)}%")

    return await fetch_character_term_ids(query_idxs, like_patterns)


async def _resolve_exact_exclude_title_term_ids(title_tokens: list[str]) -> list[int]:
    """
    Resolve EXCLUDE title tokens with exact lookup + max_df filtering.

    EXCLUDE title tokens intentionally skip expansion. Each token is
    resolved with an exact title-token lookup that still enforces MAX_DF.

    Args:
        title_tokens: Normalized title tokens to exclude.

    Returns:
        Deduplicated term_ids preserving first-seen order.
    """
    if not title_tokens:
        return []

    resolved = await fetch_title_token_ids_exact(title_tokens, max_df=MAX_DF)
    term_ids: list[int] = []
    seen_ids: set[int] = set()
    for token_idx in range(len(title_tokens)):
        token_ids = resolved.get(token_idx, [])
        for term_id in token_ids:
            if term_id in seen_ids:
                continue
            seen_ids.add(term_id)
            term_ids.append(term_id)
    return term_ids


# ═══════════════════════════════════════════════════════════════════════════════
#                     Inverted Index Posting Searches
# ═══════════════════════════════════════════════════════════════════════════════

async def search_people_postings(
    people_term_ids: list[int],
    filters: Optional[MetadataFilters] = None,
    exclude_movie_ids: Optional[set[int]] = None,
) -> dict[int, int]:
    """
    Count distinct matched people per movie.

    Args:
        people_term_ids: Resolved string_ids for INCLUDE people phrases.
        filters:         Optional metadata hard-filters.
        exclude_movie_ids: Optional movie IDs that should be excluded.

    Returns:
        Dict of {movie_id: matched_people_count}.
    """
    return await fetch_phrase_postings_match_counts(
        PostingTable.PERSON, people_term_ids, filters, exclude_movie_ids,
    )

async def search_studio_postings(
    studio_term_ids: list[int],
    filters: Optional[MetadataFilters] = None,
    exclude_movie_ids: Optional[set[int]] = None,
) -> dict[int, int]:
    """
    Count distinct matched studios per movie.

    Args:
        studio_term_ids: Resolved string_ids for INCLUDE studio phrases.
        filters:         Optional metadata hard-filters.
        exclude_movie_ids: Optional movie IDs that should be excluded.

    Returns:
        Dict of {movie_id: matched_studio_count}.
    """
    return await fetch_phrase_postings_match_counts(
        PostingTable.STUDIO, studio_term_ids, filters, exclude_movie_ids,
    )


async def search_character_postings(
    character_phrases: list[str],
    filters: Optional[MetadataFilters] = None,
    exclude_movie_ids: Optional[set[int]] = None,
) -> dict[int, int]:
    """
    Count distinct matched character query phrases per movie.

    Calls _resolve_character_term_ids to get substring-expanded term_ids,
    then runs the standard posting join pattern: unnest (query_idx, term_id)
    pairs, join inv_character_postings, COUNT(DISTINCT query_idx) per movie.

    Args:
        character_phrases: Normalized character phrase strings.
        filters:           Optional metadata hard-filters.
        exclude_movie_ids: Optional movie IDs that should be excluded.

    Returns:
        Dict of {movie_id: matched_character_count}.
    """
    if not character_phrases:
        return {}

    # Phase 1: resolve phrases → term_ids
    phrase_term_map = await _resolve_character_term_ids(character_phrases)

    if not phrase_term_map:
        return {}

    # Flatten into parallel arrays for unnest
    query_idxs: list[int] = []
    term_ids: list[int] = []
    for q_idx, tids in phrase_term_map.items():
        for tid in tids:
            query_idxs.append(q_idx)
            term_ids.append(tid)

    if not term_ids:
        return {}

    # Phase 2: posting join
    use_eligible = filters is not None and filters.is_active

    return await fetch_character_match_counts(
        query_idxs,
        term_ids,
        use_eligible,
        filters,
        exclude_movie_ids,
    )


async def search_character_postings_by_query(
    character_phrases: list[str],
    filters: Optional[MetadataFilters] = None,
    exclude_movie_ids: Optional[set[int]] = None,
) -> list[dict[int, int]]:
    """
    Return per-query character match maps from one batched DB query.

    This keeps per-franchise scoring exact while avoiding one round-trip per
    phrase. The list index aligns with character_phrases order.

    Args:
        character_phrases: Normalized character phrase strings.
        filters: Optional metadata hard-filters.
        exclude_movie_ids: Optional movie IDs that should be excluded.

    Returns:
        List of dicts where each dict is {movie_id: matched_count} for the
        phrase at that index.
    """
    if not character_phrases:
        return []

    phrase_term_map = await _resolve_character_term_ids(character_phrases)
    if not phrase_term_map:
        return [{} for _ in character_phrases]

    query_idxs: list[int] = []
    term_ids: list[int] = []
    for q_idx, tids in phrase_term_map.items():
        for term_id in tids:
            query_idxs.append(q_idx)
            term_ids.append(term_id)

    if not term_ids:
        return [{} for _ in character_phrases]

    use_eligible = filters is not None and filters.is_active
    by_query = await fetch_character_match_counts_by_query(
        query_idxs,
        term_ids,
        use_eligible,
        filters,
        exclude_movie_ids,
    )
    return [by_query.get(idx, {}) for idx in range(len(character_phrases))]


async def search_title_postings(
    token_term_id_map: dict[int, list[int]],
    filters: Optional[MetadataFilters] = None,
    exclude_movie_ids: Optional[set[int]] = None,
) -> dict[int, float]:
    """
    Compute title F-scores for one title search by joining token postings.

    Builds a MATERIALIZED eligible CTE when metadata filters are active
    (avoiding large array transfer), counts matched token positions (m)
    per movie, then applies the coverage-weighted F-score:

        title_score = (1+β²)·(coverage·specificity) / (β²·specificity + coverage)

    where coverage = m/k, specificity = m/L, β = TITLE_SCORE_BETA.

    Results are capped at TITLE_MAX_CANDIDATES (sorted by score desc) in
    the SQL query itself for wire-transfer efficiency.

    Args:
        token_term_id_map: Mapping of token_idx → list[term_id].
            Each key is a positional index for a query token; values are
            all candidate string_ids that matched that position.
        filters: Optional metadata hard-filters.  When active, an
            eligible-set CTE is built inline rather than passing IDs.
        exclude_movie_ids: Optional movie IDs that should be excluded.

    Returns:
        Dict of {movie_id: title_score} for qualifying movies, capped
        at TITLE_MAX_CANDIDATES entries.
    """
    if not token_term_id_map:
        return {}

    # ── Flatten the map into parallel arrays for unnest-based CTE ─────────
    token_idxs: list[int] = []
    term_ids: list[int] = []
    for token_idx, tids in token_term_id_map.items():
        for tid in tids:
            token_idxs.append(token_idx)
            term_ids.append(tid)

    if not term_ids:
        return {}

    # Precompute F-score coefficients
    f_coeff = 1.0 + (TITLE_SCORE_BETA ** 2)
    non_empty_token_count = sum(1 for tids in token_term_id_map.values() if tids)
    if non_empty_token_count == 0:
        return {}

    # ── Dynamically build query pieces ────────────────────────────────────
    use_eligible = filters is not None and filters.is_active

    return await fetch_title_token_match_scores(
        token_idxs=token_idxs,
        term_ids=term_ids,
        use_eligible=use_eligible, 
        f_coeff=f_coeff, 
        k=non_empty_token_count,
        beta=TITLE_SCORE_BETA, 
        title_score_threshold=TITLE_SCORE_THRESHOLD, 
        title_max_candidates=TITLE_MAX_CANDIDATES, 
        filters=filters,
        exclude_movie_ids=exclude_movie_ids,
    )
# ═══════════════════════════════════════════════════════════════════════════════
#            Full Lexical Search Orchestration (Lexical Guide §10)
# ═══════════════════════════════════════════════════════════════════════════════


async def lexical_search(
    entities: ExtractedEntitiesResponse,
    filters: Optional[MetadataFilters] = None,
) -> list[LexicalCandidate]:
    """
    Full lexical search combining all buckets with OR semantics.

    Entity inputs are treated as non-normalized strings. Phrase buckets are
    normalized via normalize_string while title-space searches use
    tokenize_title_phrase.

    Franchise INCLUDE entities search both title and character spaces and
    contribute max(title_score, character_score) per franchise.

    Returns:
        List of LexicalCandidate objects sorted by normalized_lexical_score desc.
    """
    include_people: list[str] = []
    include_characters: list[str] = []
    include_studios: list[str] = []
    include_title_searches: list[list[str]] = []
    include_franchise_phrases: list[str] = []
    include_franchise_title_searches: list[list[str]] = []

    exclude_people: list[str] = []
    exclude_characters: list[str] = []
    exclude_studios: list[str] = []
    exclude_title_tokens: list[str] = []
    exclude_franchise_phrases: list[str] = []
    exclude_franchise_title_tokens: list[str] = []

    # STEP 1 - Build include / exclude lists
    for entity in entities.entity_candidates:
        source = entity.corrected_and_normalized_entity or entity.candidate_entity_phrase
        category = entity.most_likely_category
        should_exclude = entity.exclude_from_results

        if category == EntityCategory.MOVIE_TITLE.value:
            title_tokens = tokenize_title_phrase(source)
            if not title_tokens:
                continue
            if should_exclude:
                exclude_title_tokens.extend(title_tokens)
            else:
                include_title_searches.append(title_tokens)
            continue

        if category == EntityCategory.FRANCHISE.value:
            franchise_phrase = normalize_string(source)
            if not franchise_phrase:
                continue
            if should_exclude:
                exclude_franchise_phrases.append(franchise_phrase)
                exclude_franchise_title_tokens.extend(tokenize_title_phrase(franchise_phrase))
            else:
                include_franchise_phrases.append(franchise_phrase)
            continue

        # If we reach here then that means the category is CHARACTER, PERSON, or STUDIO
        phrase = normalize_string(source)
        if not phrase:
            continue

        if category == EntityCategory.PERSON.value:
            (exclude_people if should_exclude else include_people).append(phrase)
        elif category == EntityCategory.CHARACTER.value:
            (exclude_characters if should_exclude else include_characters).append(phrase)
        elif category == EntityCategory.STUDIO.value:
            (exclude_studios if should_exclude else include_studios).append(phrase)

    # STEP 2 - Deduplicate lists
    include_people = _dedupe_preserve_order(include_people)
    include_characters = _dedupe_preserve_order(include_characters)
    include_studios = _dedupe_preserve_order(include_studios)
    include_franchise_phrases = _dedupe_preserve_order(include_franchise_phrases)
    include_franchise_title_searches = [
        tokenize_title_phrase(phrase) for phrase in include_franchise_phrases
    ]
    include_franchise_title_searches = [tokens for tokens in include_franchise_title_searches if tokens]
    include_title_searches = list(dict.fromkeys(tuple(tokens) for tokens in include_title_searches))
    include_title_searches = [list(tokens) for tokens in include_title_searches if tokens]

    exclude_people = _dedupe_preserve_order(exclude_people)
    exclude_characters = _dedupe_preserve_order(exclude_characters)
    exclude_studios = _dedupe_preserve_order(exclude_studios)
    exclude_title_tokens = _dedupe_preserve_order(exclude_title_tokens)
    exclude_franchise_phrases = _dedupe_preserve_order(exclude_franchise_phrases)
    exclude_franchise_title_tokens = _dedupe_preserve_order(exclude_franchise_title_tokens)

    # STEP 3 - Calculate max possible score
    max_possible = (
        len(include_people)
        + len(include_characters)
        + len(include_studios)
        + len(include_title_searches)
        + len(include_franchise_phrases)
    )
    if max_possible == 0:
        return []

    # STEP 4 - Resolve all include/exclude IDs (phrases + title tokens + character terms)
    exclude_character_phrases = _dedupe_preserve_order(
        exclude_characters + exclude_franchise_phrases
    )
    all_exact_phrases = _dedupe_preserve_order(
        include_people
        + include_studios
        + exclude_people
        + exclude_studios
    )
    all_exclude_title_tokens = _dedupe_preserve_order(
        exclude_title_tokens + exclude_franchise_title_tokens
    )
    all_title_searches = include_title_searches + include_franchise_title_searches
    phrase_id_map, all_title_token_maps, exclude_character_term_map, exclude_title_term_ids = await asyncio.gather(
        fetch_phrase_term_ids(all_exact_phrases),
        _resolve_all_title_tokens(all_title_searches),
        _resolve_character_term_ids(exclude_character_phrases),
        _resolve_exact_exclude_title_term_ids(all_exclude_title_tokens),
    )

    # We want to separate regular title searches from franchise title searches (scoring for franchises is unique)
    regular_title_count = len(include_title_searches)
    title_token_maps = all_title_token_maps[:regular_title_count]
    franchise_title_token_maps = all_title_token_maps[regular_title_count:]

    people_term_ids = [phrase_id_map[p] for p in include_people if p in phrase_id_map]
    studio_term_ids = [phrase_id_map[s] for s in include_studios if s in phrase_id_map]
    exclude_people_ids = [phrase_id_map[p] for p in exclude_people if p in phrase_id_map]
    exclude_studio_ids = [phrase_id_map[s] for s in exclude_studios if s in phrase_id_map]
    exclude_character_term_ids = list({
        term_id
        for term_ids in exclude_character_term_map.values()
        for term_id in term_ids
    })

    # STEP 5 - Resolve a global excluded movie-id set across all exclude buckets.
    excluded_movie_ids: set[int] = set()
    exclusion_resolution_tasks = []
    if exclude_people_ids:
        exclusion_resolution_tasks.append(
            fetch_movie_ids_by_term_ids(PostingTable.PERSON, exclude_people_ids)
        )
    if exclude_studio_ids:
        exclusion_resolution_tasks.append(
            fetch_movie_ids_by_term_ids(PostingTable.STUDIO, exclude_studio_ids)
        )
    if exclude_character_term_ids:
        exclusion_resolution_tasks.append(
            fetch_movie_ids_by_term_ids(PostingTable.CHARACTER, exclude_character_term_ids)
        )
    if exclude_title_term_ids:
        exclusion_resolution_tasks.append(
            fetch_movie_ids_by_term_ids(PostingTable.TITLE_TOKEN, exclude_title_term_ids)
        )

    if exclusion_resolution_tasks:
        exclusion_results = await asyncio.gather(*exclusion_resolution_tasks)
        for excluded_ids in exclusion_results:
            excluded_movie_ids.update(excluded_ids)

    # STEP 6 - Match scoring
    title_futures = [
        search_title_postings(token_map, filters, excluded_movie_ids)
        for token_map in title_token_maps
    ]
    franchise_title_futures = [
        search_title_postings(token_map, filters, excluded_movie_ids)
        for token_map in franchise_title_token_maps
    ]
    all_results = await asyncio.gather(
        search_people_postings(people_term_ids, filters, excluded_movie_ids),
        search_character_postings(include_characters, filters, excluded_movie_ids),
        search_studio_postings(studio_term_ids, filters, excluded_movie_ids),
        search_character_postings_by_query(
            include_franchise_phrases,
            filters,
            excluded_movie_ids,
        ),
        *title_futures,
        *franchise_title_futures,
    )

    people_scores: dict[int, int] = all_results[0]
    character_scores: dict[int, int] = all_results[1]
    studio_scores: dict[int, int] = all_results[2]

    franchise_character_results: list[dict[int, int]] = all_results[3]
    result_idx = 4
    title_score_results: list[dict[int, float]] = list(
        all_results[result_idx: result_idx + len(title_futures)]
    )
    result_idx += len(title_futures)
    franchise_title_results: list[dict[int, float]] = list(
        all_results[result_idx: result_idx + len(franchise_title_futures)]
    )
    title_score_sums: dict[int, float] = {}
    for title_result in title_score_results:
        for movie_id, score in title_result.items():
            title_score_sums[movie_id] = title_score_sums.get(movie_id, 0.0) + score

    if len(title_score_sums) > TITLE_MAX_CANDIDATES:
        sorted_titles = sorted(title_score_sums.items(), key=lambda x: x[1], reverse=True)
        title_score_sums = dict(sorted_titles[:TITLE_MAX_CANDIDATES])

    franchise_score_sums: dict[int, float] = {}
    for title_map, character_map in zip(franchise_title_results, franchise_character_results):
        franchise_movie_ids = set(title_map.keys()) | set(character_map.keys())
        for movie_id in franchise_movie_ids:
            title_score = title_map.get(movie_id, 0.0)
            character_score = float(character_map.get(movie_id, 0))
            best_score = max(title_score, character_score)
            if best_score > 0.0:
                franchise_score_sums[movie_id] = (
                    franchise_score_sums.get(movie_id, 0.0) + best_score
                )

    # STEP 7 - Build candidates (exclusions already applied in posting searches)
    all_movie_ids: set[int] = set()
    all_movie_ids.update(people_scores.keys())
    all_movie_ids.update(character_scores.keys())
    all_movie_ids.update(studio_scores.keys())
    all_movie_ids.update(title_score_sums.keys())
    all_movie_ids.update(franchise_score_sums.keys())
    candidates: list[LexicalCandidate] = []
    for movie_id in all_movie_ids:
        matched_people = people_scores.get(movie_id, 0)
        matched_characters = character_scores.get(movie_id, 0)
        matched_studios = studio_scores.get(movie_id, 0)
        title_sum = title_score_sums.get(movie_id, 0.0)
        franchise_sum = franchise_score_sums.get(movie_id, 0.0)
        raw_score = (
            float(matched_people)
            + float(matched_characters)
            + float(matched_studios)
            + title_sum
            + franchise_sum
        )

        candidates.append(
            LexicalCandidate(
                movie_id=movie_id,
                matched_people_count=matched_people,
                matched_character_count=matched_characters,
                matched_studio_count=matched_studios,
                title_score_sum=title_sum,
                franchise_score_sum=franchise_sum,
                normalized_lexical_score=raw_score / float(max_possible),
            )
        )

    candidates.sort(key=lambda c: c.normalized_lexical_score, reverse=True)
    return candidates