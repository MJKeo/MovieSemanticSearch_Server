import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional, Tuple


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
    ReceptionType,
    BudgetSize,
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
    LanguageListPreference,
    WatchProvidersPreference,
    MaturityPreference,
    PopularTrendingPreference,
    ReceptionPreference,
    BudgetSizePreference,
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
    MetadataPreferenceName.AUDIO_LANGUAGES: (LanguageListPreference, "audio_languages_preference"),
    MetadataPreferenceName.WATCH_PROVIDERS: (WatchProvidersPreference, "watch_providers_preference"),
    MetadataPreferenceName.MATURITY_RATING: (MaturityPreference, "maturity_rating_preference"),
    MetadataPreferenceName.POPULARITY: (PopularTrendingPreference, "popular_trending_preference"),
    MetadataPreferenceName.RECEPTION: (ReceptionPreference, "reception_preference"),
    MetadataPreferenceName.BUDGET_SIZE: (BudgetSizePreference, "budget_size_preference"),
}

_METADATA_PREFERENCE_DEFAULT_FACTORIES: dict[MetadataPreferenceName, Callable[[], Any]] = {
    MetadataPreferenceName.RELEASE_DATE: lambda: DatePreference(result=None),
    MetadataPreferenceName.DURATION: lambda: NumericalPreference(result=None),
    MetadataPreferenceName.GENRES: lambda: GenreListPreference(result=None),
    MetadataPreferenceName.AUDIO_LANGUAGES: lambda: LanguageListPreference(result=None),
    MetadataPreferenceName.WATCH_PROVIDERS: lambda: WatchProvidersPreference(result=None),
    MetadataPreferenceName.MATURITY_RATING: lambda: MaturityPreference(result=None),
    MetadataPreferenceName.POPULARITY: lambda: PopularTrendingPreference(
        prefers_trending_movies=False,
        prefers_popular_movies=False,
    ),
    MetadataPreferenceName.RECEPTION: lambda: ReceptionPreference(
        reception_type=ReceptionType.NO_PREFERENCE,
    ),
    MetadataPreferenceName.BUDGET_SIZE: lambda: BudgetSizePreference(budget_size=BudgetSize.NO_PREFERENCE),
}


def _build_default_metadata_preference(preference_name: MetadataPreferenceName) -> Any:
    """Create the safe fallback response for a single metadata preference."""
    return _METADATA_PREFERENCE_DEFAULT_FACTORIES[preference_name]()


# ===============================
#           Lexical
# ===============================

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

async def create_channel_weights_async(query) -> Optional[ChannelWeightsResponse]:
    """
        Extract channel weights (lexical, metadata, vector relevance) from query.
        Throws an error if anything fails.
    """
    return await generate_kimi_response_async(
        user_prompt=f"User query: \"{query}\"",
        system_prompt=CHANNEL_WEIGHTS_SYSTEM_PROMPT,
        response_format=ChannelWeightsResponse,
    )


# ===============================
#      Metadata Preferences
# ===============================

async def extract_single_metadata_preference_async(query: str, preference_name: MetadataPreferenceName) -> Optional[Any]:
    """
    Async version of extract_single_metadata_preference.
    Extract a single metadata preference from the query using the LLM.
    Returns the parsed schema or None if extraction fails.
    """
    response_schema, _ = _METADATA_PREFERENCE_TO_RESPONSE_MAPPING[preference_name]
    system_prompt = ALL_METADATA_EXTRACTION_PROMPTS[preference_name]
    return await generate_kimi_response_async(
        user_prompt=f"User query: \"{query}\"",
        system_prompt=system_prompt,
        response_format=response_schema,
    )


async def extract_all_metadata_preferences_async(query: str) -> MetadataPreferencesResponse:
    """
    Run extract_single_metadata_preference_async concurrently for each metadata preference,
    then assemble a MetadataPreferencesResponse.

    If any individual extraction raises an exception or returns None, that preference
    is replaced with its default "no preference" value while the remaining results
    are preserved.
    """
    preferences = tuple(_METADATA_PREFERENCE_TO_RESPONSE_MAPPING)
    tasks = [
        extract_single_metadata_preference_async(query, preference_name)
        for preference_name in preferences
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    kwargs: dict[str, Any] = {}
    for preference_name, raw_result in zip(preferences, raw_results):
        _, field_name = _METADATA_PREFERENCE_TO_RESPONSE_MAPPING[preference_name]

        if isinstance(raw_result, Exception):
            kwargs[field_name] = _build_default_metadata_preference(preference_name)
            continue

        if raw_result is None:
            kwargs[field_name] = _build_default_metadata_preference(preference_name)
            continue

        kwargs[field_name] = raw_result

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