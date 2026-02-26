from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Tuple

from implementation.llms.generic_methods import generate_kimi_response, generate_kimi_response_async
from implementation.prompts.lexical_prompts import EXTRACT_LEXICAL_ENTITIES_SYSTEM_PROMPT
from implementation.prompts.channel_weights_prompts import CHANNEL_WEIGHTS_SYSTEM_PROMPT
from implementation.prompts.metadata_preferences_prompts import ALL_METADATA_EXTRACTION_PROMPTS
from implementation.prompts.vector_subquery_prompts import VECTOR_SUBQUERY_SYSTEM_PROMPTS
from implementation.prompts.vector_weights_prompts import VECTOR_WEIGHT_SYSTEM_PROMPTS
from implementation.classes.enums import (
    VectorCollectionName,
    VectorName,
    MetadataPreferenceName,
)
from implementation.classes.schemas import (
    ExtractedEntitiesResponse,
    ChannelWeightsResponse,
    MetadataPreferencesResponse,
    VectorSubqueriesResponse,
    VectorCollectionSubqueryData,
    VectorCollectionWeightData,
    VectorWeightsResponse,
    QueryUnderstandingResponse,
    DatePreference,
    NumericalPreference,
    GenreListPreference,
    ListPreference,
    WatchProvidersPreference,
    MaturityPreference,
    PopularTrendingPreference,
    ReceptionPreference,
)

# ===============================
#           Helpers
# ===============================

# Maps each vector collection enum to its corresponding field in VectorSubqueriesResponse.
# Excludes DENSE_ANCHOR_VECTORS since it has no subquery prompt.
_COLLECTION_TO_RESPONSE_FIELD = {
    VectorCollectionName.PLOT_EVENTS_VECTORS: "plot_events_data",
    VectorCollectionName.PLOT_ANALYSIS_VECTORS: "plot_analysis_data",
    VectorCollectionName.VIEWER_EXPERIENCE_VECTORS: "viewer_experience_data",
    VectorCollectionName.WATCH_CONTEXT_VECTORS: "watch_context_data",
    VectorCollectionName.NARRATIVE_TECHNIQUES_VECTORS: "narrative_techniques_data",
    VectorCollectionName.PRODUCTION_VECTORS: "production_data",
    VectorCollectionName.RECEPTION_VECTORS: "reception_data",
}

# Maps each metadata preference enum to (BaseModel response schema, response field name).
_METADATA_PREFERENCE_TO_RESPONSE_MAPPING: dict[MetadataPreferenceName, tuple[type, str]] = {
    MetadataPreferenceName.RELEASE_DATE: (DatePreference, "release_date_preference"),
    MetadataPreferenceName.DURATION: (NumericalPreference, "duration_preference"),
    MetadataPreferenceName.GENRES: (GenreListPreference, "genres_preference"),
    MetadataPreferenceName.AUDIO_LANGUAGES: (ListPreference, "audio_languages_preference"),
    MetadataPreferenceName.WATCH_PROVIDERS: (WatchProvidersPreference, "watch_providers_preference"),
    MetadataPreferenceName.MATURITY_RATING: (MaturityPreference, "maturity_rating_preference"),
    MetadataPreferenceName.POPULARITY: (PopularTrendingPreference, "popular_trending_preference"),
    MetadataPreferenceName.RECEPTION: (ReceptionPreference, "reception_preference"),
}


# ===============================
#           Lexical
# ===============================

def extract_lexical_entities(query: str) -> Optional[ExtractedEntitiesResponse]:
    """
        Extract lexical entities (characters, franchises, people, studios, titles) from query.
        Throws an error if anything fails.
    """
    return generate_kimi_response(
        user_prompt=f"User query: \"{query}\"",
        system_prompt=EXTRACT_LEXICAL_ENTITIES_SYSTEM_PROMPT,
        response_format=ExtractedEntitiesResponse,
    )


async def extract_lexical_entities_async(query: str) -> Optional[ExtractedEntitiesResponse]:
    """
        Async version of extract_lexical_entities.
        Extract lexical entities (characters, franchises, people, studios, titles) from query.
    """
    return await generate_kimi_response_async(
        user_prompt=f'User query: "{query}"',
        system_prompt=EXTRACT_LEXICAL_ENTITIES_SYSTEM_PROMPT,
        response_format=ExtractedEntitiesResponse,
    )


# ===============================
#        Channel Weights
# ===============================

def create_channel_weights(query) -> Optional[ChannelWeightsResponse]:
    """
        Extract channel weights (lexical, metadata, vector relevance) from query.
        Throws an error if anything fails.
    """
    return generate_kimi_response(
        user_prompt=f"User query: \"{query}\"",
        system_prompt=CHANNEL_WEIGHTS_SYSTEM_PROMPT,
        response_format=ChannelWeightsResponse,
    )


# ===============================
#      Metadata Preferences
# ===============================

def extract_single_metadata_preference(query: str, preference_name: MetadataPreferenceName) -> Optional[Any]:
    """
    Extract a single metadata preference from the query using the LLM.
    Returns the parsed schema or None if extraction fails.
    """
    response_schema, _ = _METADATA_PREFERENCE_TO_RESPONSE_MAPPING[preference_name]
    system_prompt = ALL_METADATA_EXTRACTION_PROMPTS[preference_name]
    return generate_kimi_response(
        user_prompt=f"User query: \"{query}\"",
        system_prompt=system_prompt,
        response_format=response_schema,
    )


def extract_all_metadata_preferences(query: str) -> Optional[MetadataPreferencesResponse]:
    """
    Run extract_single_metadata_preference in parallel for each metadata preference,
    then assemble a MetadataPreferencesResponse. Returns None if any
    metadata preference extraction fails or returns None.
    """
    results: dict[MetadataPreferenceName, Any] = {}

    with ThreadPoolExecutor(max_workers=len(_METADATA_PREFERENCE_TO_RESPONSE_MAPPING)) as executor:
        future_to_preference = {
            executor.submit(extract_single_metadata_preference, query, pref): pref
            for pref in _METADATA_PREFERENCE_TO_RESPONSE_MAPPING
        }
        for future in as_completed(future_to_preference):
            preference = future_to_preference[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"Error extracting metadata preference {preference} for query {query}: {e}")
                return None
            if result is None:
                return None
            results[preference] = result

    # Build MetadataPreferencesResponse from results, using mapping for field names.
    kwargs: dict[str, Any] = {}
    for pref, (_, field_name) in _METADATA_PREFERENCE_TO_RESPONSE_MAPPING.items():
        kwargs[field_name] = results[pref]

    return MetadataPreferencesResponse(**kwargs)


# ===============================
#      Vector Subqueries
# ===============================

async def create_single_vector_subquery_async(
    query: str, collection_name: VectorName
) -> Optional[VectorCollectionSubqueryData]:
    """Create a single vector subquery (plot events, plot analysis, viewer experience, watch context, narrative techniques, production, reception) from query."""
    return await generate_kimi_response_async(
        user_prompt=f'User query: "{query}"',
        system_prompt=VECTOR_SUBQUERY_SYSTEM_PROMPTS[collection_name],
        response_format=VectorCollectionSubqueryData,
    )

# def create_all_vector_subqueries(query: str) -> Optional[VectorSubqueriesResponse]:
#     """
#     Run create_single_vector_subquery in parallel for each vector collection, then assemble
#     a VectorSubqueriesResponse. Returns None if any single subquery fails.
#     """
#     results: dict[VectorCollectionName, Optional[VectorCollectionSubqueryData]] = {}

#     with ThreadPoolExecutor(max_workers=len(_COLLECTION_TO_RESPONSE_FIELD)) as executor:
#         future_to_collection = {
#             executor.submit(create_single_vector_subquery, query, collection): collection
#             for collection in _COLLECTION_TO_RESPONSE_FIELD
#         }
#         for future in as_completed(future_to_collection):
#             collection = future_to_collection[future]
#             try:
#                 result = future.result()
#             except Exception:
#                 return None
#             if result is None:
#                 return None
#             results[collection] = result

#     return VectorSubqueriesResponse(
#         plot_events_data=results[VectorCollectionName.PLOT_EVENTS_VECTORS],
#         plot_analysis_data=results[VectorCollectionName.PLOT_ANALYSIS_VECTORS],
#         viewer_experience_data=results[VectorCollectionName.VIEWER_EXPERIENCE_VECTORS],
#         watch_context_data=results[VectorCollectionName.WATCH_CONTEXT_VECTORS],
#         narrative_techniques_data=results[VectorCollectionName.NARRATIVE_TECHNIQUES_VECTORS],
#         production_data=results[VectorCollectionName.PRODUCTION_VECTORS],
#         reception_data=results[VectorCollectionName.RECEPTION_VECTORS],
#     )


# ===============================
#        Vector Weights
# ===============================

async def create_single_vector_weight_async(query: str, collection_name: VectorName) -> Optional[VectorCollectionWeightData]:
    """Create a single vector weight (plot events, plot analysis, viewer experience, watch context, narrative techniques, production, reception) from query."""
    return await generate_kimi_response_async(
        user_prompt=f'User query: "{query}"',
        system_prompt=VECTOR_WEIGHT_SYSTEM_PROMPTS[collection_name],
        response_format=VectorCollectionWeightData,
    )

# def create_all_vector_weights(query: str) -> Optional[VectorWeightsResponse]:
#     """
#     Run create_single_vector_weight in parallel for each vector collection, then assemble
#     a VectorWeightsResponse. Returns None if any single weight creation fails.
#     """
#     results: dict[VectorCollectionName, Optional[VectorCollectionWeightData]] = {}

#     with ThreadPoolExecutor(max_workers=len(_COLLECTION_TO_RESPONSE_FIELD)) as executor:
#         future_to_collection = {
#             executor.submit(create_single_vector_weight, query, collection): collection
#             for collection in _COLLECTION_TO_RESPONSE_FIELD
#         }
#         for future in as_completed(future_to_collection):
#             collection = future_to_collection[future]
#             try:
#                 result = future.result()
#             except Exception:
#                 return None
#             if result is None:
#                 return None
#             results[collection] = result

#     return VectorWeightsResponse(
#         plot_events_data=results[VectorCollectionName.PLOT_EVENTS_VECTORS],
#         plot_analysis_data=results[VectorCollectionName.PLOT_ANALYSIS_VECTORS],
#         viewer_experience_data=results[VectorCollectionName.VIEWER_EXPERIENCE_VECTORS],
#         watch_context_data=results[VectorCollectionName.WATCH_CONTEXT_VECTORS],
#         narrative_techniques_data=results[VectorCollectionName.NARRATIVE_TECHNIQUES_VECTORS],
#         production_data=results[VectorCollectionName.PRODUCTION_VECTORS],
#         reception_data=results[VectorCollectionName.RECEPTION_VECTORS],
#     )


# ===============================
#   Overall Query Understanding
# ===============================

def create_overall_query_understanding(query: str) -> Optional[QueryUnderstandingResponse]:
    """
    Run all 24 query-understanding tasks in parallel (lexical, channel weights, 8 metadata
    preferences, 7 vector subqueries, 7 vector weights), then assemble QueryUnderstandingResponse.
    Returns None if: lexical_entities, channel_weights, any metadata preference, any vector
    subquery, any vector weight fails or returns None; or if any exception occurs.
    """
    # Build all 24 tasks: 1 lexical + 1 channel_weights + 8 metadata + 7 subqueries + 7 weights
    future_to_key: dict = {}
    total_tasks = (
        1  # lexical
        + 1  # channel_weights
        + len(_METADATA_PREFERENCE_TO_RESPONSE_MAPPING)
        + len(_COLLECTION_TO_RESPONSE_FIELD)
        + len(_COLLECTION_TO_RESPONSE_FIELD)
    )

    with ThreadPoolExecutor(max_workers=total_tasks) as executor:
        future_to_key[executor.submit(extract_lexical_entities, query)] = ("lexical", None)
        future_to_key[executor.submit(create_channel_weights, query)] = ("channel_weights", None)

        for pref in _METADATA_PREFERENCE_TO_RESPONSE_MAPPING:
            future_to_key[executor.submit(extract_single_metadata_preference, query, pref)] = (
                "metadata",
                pref,
            )

        for coll in _COLLECTION_TO_RESPONSE_FIELD:
            future_to_key[executor.submit(create_single_vector_subquery_async, query, coll)] = (
                "subquery",
                coll,
            )

        for coll in _COLLECTION_TO_RESPONSE_FIELD:
            future_to_key[executor.submit(create_single_vector_weight_async, query, coll)] = (
                "weight",
                coll,
            )

        results: dict = {}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result = future.result()
                task_type, identifier = key

                # These tasks must succeed; None means failure
                if task_type == "lexical" or task_type == "channel_weights":
                    if result is None:
                        return None
                elif task_type == "subquery" or task_type == "weight":
                    if result is None:
                        return None
                elif task_type == "metadata":
                    if result is None:
                        return None

                results[key] = result
            except Exception:
                return None

    # Assemble MetadataPreferencesResponse from metadata results.
    # All metadata preferences are required and already validated as non-None.
    metadata_kwargs: dict[str, Any] = {}
    for pref, (_, field_name) in _METADATA_PREFERENCE_TO_RESPONSE_MAPPING.items():
        metadata_kwargs[field_name] = results[("metadata", pref)]

    return QueryUnderstandingResponse(
        channel_weights=results[("channel_weights", None)],
        lexical_entities=results[("lexical", None)],
        metadata_preferences=MetadataPreferencesResponse(**metadata_kwargs),
        vector_subqueries=VectorSubqueriesResponse(
            plot_events_data=results[("subquery", VectorCollectionName.PLOT_EVENTS_VECTORS)],
            plot_analysis_data=results[("subquery", VectorCollectionName.PLOT_ANALYSIS_VECTORS)],
            viewer_experience_data=results[("subquery", VectorCollectionName.VIEWER_EXPERIENCE_VECTORS)],
            watch_context_data=results[("subquery", VectorCollectionName.WATCH_CONTEXT_VECTORS)],
            narrative_techniques_data=results[("subquery", VectorCollectionName.NARRATIVE_TECHNIQUES_VECTORS)],
            production_data=results[("subquery", VectorCollectionName.PRODUCTION_VECTORS)],
            reception_data=results[("subquery", VectorCollectionName.RECEPTION_VECTORS)],
        ),
        vector_weights=VectorWeightsResponse(
            plot_events_data=results[("weight", VectorCollectionName.PLOT_EVENTS_VECTORS)],
            plot_analysis_data=results[("weight", VectorCollectionName.PLOT_ANALYSIS_VECTORS)],
            viewer_experience_data=results[("weight", VectorCollectionName.VIEWER_EXPERIENCE_VECTORS)],
            watch_context_data=results[("weight", VectorCollectionName.WATCH_CONTEXT_VECTORS)],
            narrative_techniques_data=results[("weight", VectorCollectionName.NARRATIVE_TECHNIQUES_VECTORS)],
            production_data=results[("weight", VectorCollectionName.PRODUCTION_VECTORS)],
            reception_data=results[("weight", VectorCollectionName.RECEPTION_VECTORS)],
        ),
    )