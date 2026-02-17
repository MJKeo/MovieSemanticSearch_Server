# QueryUnderstandingResponse Schema Reference

This document provides a complete reference for the `QueryUnderstandingResponse` schema, which represents the structured output of our query understanding pipeline. When a user enters a movie search query, this schema captures everything we need to route the query to the appropriate search channels and execute an effective multi-channel retrieval.

## Table of Contents

1. [Overview](#overview)
2. [Top-Level Structure](#top-level-structure)
3. [Channel Weights](#channel-weights-channelweightsresponse)
4. [Lexical Entities](#lexical-entities-extractedentitiesresponse)
5. [Metadata Preferences](#metadata-preferences-metadatapreferencesresponse)
6. [Vector Subqueries](#vector-subqueries-vectorsubqueriesresponse)
7. [Vector Weights](#vector-weights-vectorweightsresponse)
8. [Shared Enums Reference](#shared-enums-reference)

---

## Overview

The query understanding pipeline takes a raw user search query (e.g., "90s action movies like Die Hard with practical effects") and decomposes it into structured components that can be used by different search channels:

- **Lexical Search**: Finds movies by exact entity matches (actors, directors, titles, etc.)
- **Metadata Filtering**: Filters movies by structured attributes (release date, genre, runtime, etc.)
- **Vector Search**: Finds movies by semantic similarity across multiple embedding spaces

The `QueryUnderstandingResponse` captures the output of all these extraction processes in a single unified structure.

---

## Top-Level Structure

```python
class QueryUnderstandingResponse(BaseModel):
    channel_weights: ChannelWeightsResponse
    lexical_entities: ExtractedEntitiesResponse
    metadata_preferences: MetadataPreferencesResponse
    vector_subqueries: VectorSubqueriesResponse
    vector_weights: VectorWeightsResponse
```

| Field | Type | Description |
|-------|------|-------------|
| `channel_weights` | `ChannelWeightsResponse` | Relative importance of each search channel (lexical, metadata, vector) for this query |
| `lexical_entities` | `ExtractedEntitiesResponse` | Named entities extracted from the query (people, titles, franchises, etc.) |
| `metadata_preferences` | `MetadataPreferencesResponse` | Structured filters for movie attributes (date, genre, runtime, etc.) |
| `vector_subqueries` | `VectorSubqueriesResponse` | Optimized query text for each vector collection |
| `vector_weights` | `VectorWeightsResponse` | Relevance weight for each vector collection |

---

## Channel Weights (`ChannelWeightsResponse`)

Determines the relative importance of each search channel for the given query. This helps the retrieval system allocate resources and combine results appropriately.

### Schema

```python
class ChannelWeightsResponse(BaseModel):
    lexical_relevance: RelevanceSize
    metadata_relevance: RelevanceSize
    vector_relevance: RelevanceSize
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `lexical_relevance` | `RelevanceSize` | How much of the query's intent relates to finding specific named entities |
| `metadata_relevance` | `RelevanceSize` | How much of the query's intent relates to filterable metadata attributes |
| `vector_relevance` | `RelevanceSize` | How much of the query's intent relates to semantic/conceptual matching |

### Semantic Meaning

The channel weights router analyzes what the user is searching for and estimates which channels will be most useful:

- **Lexical relevance** increases when the user mentions specific entities:
  - Character names (e.g., "James Bond", "Hannibal Lecter")
  - Franchise/series names (e.g., "Marvel", "Fast and Furious")
  - Real people (e.g., "Spielberg", "Tom Hanks")
  - Studios (e.g., "A24", "Studio Ghibli")
  - Movie titles (e.g., "Goodfellas", "The Matrix")

- **Metadata relevance** increases when the user specifies concrete attributes:
  - Release date/decade/year
  - Duration/runtime
  - Genres
  - Audio languages
  - Streaming platforms
  - Maturity rating
  - Trending/popularity status
  - Critical reception level

- **Vector relevance** increases for semantic intent:
  - Plot/story content ("heist gone wrong", "revenge story")
  - Themes and arcs ("redemption", "coming of age")
  - Viewer experience ("cozy", "intense", "heartwarming")
  - Watch context ("date night movie", "background noise")
  - Storytelling techniques ("twist ending", "unreliable narrator")

### Example

For the query "90s Spielberg movies that are heartwarming":
- `lexical_relevance`: `"medium"` (Spielberg is a named person)
- `metadata_relevance`: `"medium"` (90s is a release decade)
- `vector_relevance`: `"medium"` (heartwarming is a viewer experience)

---

## Lexical Entities (`ExtractedEntitiesResponse`)

Contains all named entities extracted from the user's query. These are used for exact-match lookups in our lexical database.

### Schema

```python
class ExtractedEntitiesResponse(BaseModel):
    entity_candidates: List[ExtractedEntityData]

class ExtractedEntityData(BaseModel):
    candidate_entity_phrase: str
    most_likely_category: EntityCategory
    exclude_from_results: bool
    corrected_and_normalized_entity: str
```

### Fields

#### `ExtractedEntitiesResponse`

| Field | Type | Description |
|-------|------|-------------|
| `entity_candidates` | `List[ExtractedEntityData]` | List of all extracted entities from the query |

#### `ExtractedEntityData`

| Field | Type | Description |
|-------|------|-------------|
| `candidate_entity_phrase` | `str` | The original verbatim word/phrase from the query |
| `most_likely_category` | `EntityCategory` | The category that best represents this entity |
| `exclude_from_results` | `bool` | `true` if user wants to EXCLUDE this entity, `false` to INCLUDE |
| `corrected_and_normalized_entity` | `str` | Minimally corrected form (typo fixes, capitalization) |

### `EntityCategory` Enum

```python
class EntityCategory(Enum):
    CHARACTER = "character"      # Fictional characters (e.g., "James Bond", "Ellen Ripley")
    FRANCHISE = "franchise"      # Multi-film series (e.g., "Marvel", "Mission Impossible")
    MOVIE_TITLE = "movie_title"  # Specific movie names (e.g., "Goodfellas", "The Matrix")
    PERSON = "person"            # Real people (e.g., "Spielberg", "Tom Hanks")
    STUDIO = "studio"            # Production companies (e.g., "A24", "Universal")
```

### Semantic Meaning

The lexical entity extractor identifies **specific named things** that the user references. It does NOT extract:
- Genre terms ("action", "comedy")
- Tone descriptors ("dark", "lighthearted")
- Plot elements ("twist ending", "heist")
- Temporal descriptors ("90s", "classic")

**Key distinction**: The user must actually TYPE the entity name. Describing a movie's plot does NOT count as naming it.

**Normalization rules**:
- Fix typos: "Johny Dep" → "Johnny Depp"
- Fix capitalization: "spielberg" → "Spielberg"
- Complete unambiguous person names: "Scorsese" → "Martin Scorsese"
- Do NOT add articles: "godfather" stays "Godfather" (not "The Godfather")
- Do NOT infer titles from descriptions

### Example

For the query "Coppola mafia movies not starring Pacino":

```json
{
  "entity_candidates": [
    {
      "candidate_entity_phrase": "Coppola",
      "most_likely_category": "person",
      "exclude_from_results": false,
      "corrected_and_normalized_entity": "Francis Ford Coppola"
    },
    {
      "candidate_entity_phrase": "Pacino",
      "most_likely_category": "person",
      "exclude_from_results": true,
      "corrected_and_normalized_entity": "Al Pacino"
    }
  ]
}
```

Note: "mafia" is NOT extracted because it's a theme/genre descriptor, not a named entity.

---

## Metadata Preferences (`MetadataPreferencesResponse`)

Contains structured filters for concrete movie attributes. Each preference is optional (can be `null` if not specified in the query).

### Schema

```python
class MetadataPreferencesResponse(BaseModel):
    release_date_preference: DatePreference
    duration_preference: NumericalPreference
    genres_preference: GenreListPreference
    audio_languages_preference: ListPreference
    watch_providers_preference: WatchProvidersPreference
    maturity_rating_preference: MaturityPreference
    popular_trending_preference: PopularTrendingPreference
    reception_preference: ReceptionPreference
```

### Fields Overview

| Field | Type | Description |
|-------|------|-------------|
| `release_date_preference` | `DatePreference` | When the movie was released |
| `duration_preference` | `NumericalPreference` | How long the movie is (in minutes) |
| `genres_preference` | `GenreListPreference` | Which genres to include/exclude |
| `audio_languages_preference` | `ListPreference` | Which audio languages to include/exclude |
| `watch_providers_preference` | `WatchProvidersPreference` | Streaming platforms and access type |
| `maturity_rating_preference` | `MaturityPreference` | Age-appropriateness rating |
| `popular_trending_preference` | `PopularTrendingPreference` | Whether user wants popular/trending movies |
| `reception_preference` | `ReceptionPreference` | Critical reception level |

---

### Release Date Preference

```python
class DatePreference(BaseModel):
    result: Optional[DatePreferenceResult]

class DatePreferenceResult(BaseModel):
    first_date: str  # ISO 8601 format: YYYY-MM-DD
    match_operation: DateMatchOperation
    second_date: Optional[str]  # Only for BETWEEN operations
```

#### `DateMatchOperation` Enum

```python
class DateMatchOperation(Enum):
    EXACT = "exact"      # Match exactly this date (rare)
    BEFORE = "before"    # Released before first_date
    AFTER = "after"      # Released after first_date
    BETWEEN = "between"  # Released between first_date and second_date
```

#### Semantic Meaning

Extracts time period preferences from the query:
- Decades use `BETWEEN`: "80s" → 1980-01-01 to 1989-12-31
- Single years use `BETWEEN`: "2015 films" → 2015-01-01 to 2015-12-31
- Relative terms: "classic" → before 1980; "recent" → after 2015
- "Retro aesthetic" is NOT a date preference (it's a visual style)

---

### Duration Preference

```python
class NumericalPreference(BaseModel):
    result: Optional[NumericalPreferenceResult]

class NumericalPreferenceResult(BaseModel):
    first_value: float  # Duration in minutes
    match_operation: NumericalMatchOperation
    second_value: Optional[float]  # Only for BETWEEN operations
```

#### `NumericalMatchOperation` Enum

```python
class NumericalMatchOperation(Enum):
    EXACT = "exact"              # Exactly this duration
    BETWEEN = "between"          # Between first_value and second_value
    LESS_THAN = "less_than"      # Shorter than first_value
    GREATER_THAN = "greater_than"  # Longer than first_value
```

#### Semantic Meaning

Extracts runtime preferences:
- "Short film" → less than 100 minutes
- "Long movie" → greater than 120 minutes
- "Around two hours" → between 110 and 130 minutes

**Important**: Pacing descriptors ("slow burn", "fast-paced") are NOT duration preferences. A 3-hour film can be fast-paced; a 90-minute film can feel slow.

---

### Genres Preference

```python
class GenreListPreference(BaseModel):
    result: Optional[GenreListPreferenceResult]

class GenreListPreferenceResult(BaseModel):
    should_include: List[Genre]  # Genres the user wants
    should_exclude: List[Genre]  # Genres the user wants to avoid
```

#### `Genre` Enum

```python
class Genre(Enum):
    ACTION = "Action"
    ADVENTURE = "Adventure"
    ANIMATION = "Animation"
    BIOGRAPHY = "Biography"
    COMEDY = "Comedy"
    CRIME = "Crime"
    DOCUMENTARY = "Documentary"
    DRAMA = "Drama"
    FAMILY = "Family"
    FANTASY = "Fantasy"
    FILM_NOIR = "Film-Noir"
    GAME_SHOW = "Game-Show"
    HISTORY = "History"
    HORROR = "Horror"
    MUSIC = "Music"
    MUSICAL = "Musical"
    MYSTERY = "Mystery"
    NEWS = "News"
    REALITY_TV = "Reality-TV"
    ROMANCE = "Romance"
    SCI_FI = "Sci-Fi"
    SHORT = "Short"
    SPORT = "Sport"
    TALK_SHOW = "Talk-Show"
    THRILLER = "Thriller"
    WAR = "War"
    WESTERN = "Western"
```

#### Semantic Meaning

Maps genre terms to our standardized genre list:
- Compound genres: "romantic comedy" → [Romance, Comedy]
- Colloquial terms: "romcom" → [Romance, Comedy]; "biopic" → [Biography]
- Subgenres map to parent: "slasher" → [Horror]; "heist" → [Crime, Thriller]

**Exclusions** require explicit rejection language ("no horror", "skip the romcoms"). Wanting Comedy does NOT imply excluding Drama.

---

### Audio Languages Preference

```python
class ListPreference(BaseModel):
    result: Optional[ListPreferenceResult]

class ListPreferenceResult(BaseModel):
    should_include: List[str]  # Languages to include
    should_exclude: List[str]  # Languages to exclude
```

#### Semantic Meaning

Extracts spoken language preferences:
- "French cinema" → include French
- "Korean audio" → include Korean
- Country mappings: "Brazilian movie" → Portuguese

**Important**: Subtitles ≠ audio language. "With English subtitles" does NOT mean English audio.

---

### Watch Providers Preference

```python
class WatchProvidersPreference(BaseModel):
    result: Optional[WatchProvidersPreferenceResult]

class WatchProvidersPreferenceResult(BaseModel):
    should_include: List[str]  # Platforms to include
    should_exclude: List[str]  # Platforms to exclude
    preferred_access_type: Optional[StreamingAccessType]
```

#### `StreamingAccessType` Enum

```python
class StreamingAccessType(Enum):
    SUBSCRIPTION = "subscription"  # Included with subscription
    RENT = "rent"                  # Available to rent
    BUY = "buy"                    # Available to purchase
```

#### Semantic Meaning

Extracts streaming platform preferences:
- Platform normalization: "Prime" → "Amazon Prime Video"; "Max" → "HBO Max"
- Access type indicators: "streaming on" → subscription; "rent" → rent

**Important**: Studios (Disney, A24) are NOT streaming platforms.

---

### Maturity Rating Preference

```python
class MaturityPreference(BaseModel):
    result: Optional[MaturityPreferenceResult]

class MaturityPreferenceResult(BaseModel):
    rating: str  # G, PG, PG-13, R, NC-17
    match_operation: RatingMatchOperation
```

#### `RatingMatchOperation` Enum

```python
class RatingMatchOperation(Enum):
    EXACT = "exact"                          # Must be exactly this rating
    GREATER_THAN = "greater_than"            # More mature than this rating
    LESS_THAN = "less_than"                  # Less mature than this rating
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
```

#### Semantic Meaning

Rating scale (least to most mature): G → PG → PG-13 → R → NC-17

- "Safe for my toddler" → G, less_than_or_equal
- "Teen-appropriate" → PG-13, less_than_or_equal
- "Adults-only" → R, greater_than_or_equal

**Important**: Content descriptors ("violent", "scary") are NOT maturity ratings. A PG-13 film can be intense.

---

### Popular/Trending Preference

```python
class PopularTrendingPreference(BaseModel):
    prefers_trending_movies: bool
    prefers_popular_movies: bool
```

#### Semantic Meaning

Two independent dimensions:
- **Trending**: Currently buzzing, what's hot RIGHT NOW ("everyone's watching", "viral")
- **Popular**: Widely known, mainstream, all-time hits ("blockbuster", "household name")

Default is `false` for both. "Hidden gems" and "cult classics" explicitly indicate NOT popular.

**Important**: Popularity ≠ critical acclaim. A blockbuster can have terrible reviews.

---

### Reception Preference

```python
class ReceptionPreference(BaseModel):
    reception_type: ReceptionType
```

#### `ReceptionType` Enum

```python
class ReceptionType(Enum):
    CRITICALLY_ACCLAIMED = "critically_acclaimed"  # Award-winning, highly rated
    POORLY_RECEIVED = "poorly_received"            # Panned, so-bad-it's-good
    NO_PREFERENCE = "no_preference"                # Default
```

#### Semantic Meaning

- **Critically acclaimed triggers**: "Oscar-winning", "masterpiece", "highest rated"
- **Poorly received triggers**: "bombed", "so bad it's good", "Razzie winners"

**Important**: Subjective quality words ("good", "great", "fun") are NOT enough. "Good thriller" → `no_preference`.

---

## Vector Subqueries (`VectorSubqueriesResponse`)

Contains optimized query text for each of our seven vector collections. Each subquery is tailored to match the content embedded in that specific vector space.

### Schema

```python
class VectorSubqueriesResponse(BaseModel):
    plot_events_data: VectorCollectionSubqueryData
    plot_analysis_data: VectorCollectionSubqueryData
    viewer_experience_data: VectorCollectionSubqueryData
    watch_context_data: VectorCollectionSubqueryData
    narrative_techniques_data: VectorCollectionSubqueryData
    production_data: VectorCollectionSubqueryData
    reception_data: VectorCollectionSubqueryData

class VectorCollectionSubqueryData(BaseModel):
    justification: str  # Brief explanation of the decision
    relevant_subquery_text: Optional[str]  # Comma-separated phrases, or null
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `plot_events_data` | `VectorCollectionSubqueryData` | Query for plot summaries (what literally happens) |
| `plot_analysis_data` | `VectorCollectionSubqueryData` | Query for themes, genres, and story types |
| `viewer_experience_data` | `VectorCollectionSubqueryData` | Query for how it feels to watch |
| `watch_context_data` | `VectorCollectionSubqueryData` | Query for why/when to watch |
| `narrative_techniques_data` | `VectorCollectionSubqueryData` | Query for storytelling craft |
| `production_data` | `VectorCollectionSubqueryData` | Query for production facts |
| `reception_data` | `VectorCollectionSubqueryData` | Query for audience reception |

### Vector Collection Descriptions

#### Plot Events
**Contains**: Dense narrative prose — chronological plot summaries with character names, specific events, story settings, character motivations, plot mechanics.

**Good for**: "heist goes wrong", "detective investigates murder", "set during WWII", "siblings reunite"

**Not good for**: Genre labels, thematic interpretations, how it feels to watch, production facts

#### Plot Analysis
**Contains**: Thematic interpretations — core concepts, genre signatures, character arcs, themes, lessons learned, generalized plot overviews.

**Good for**: "psychological thriller", "redemption story", "explores grief", "coming-of-age"

**Not good for**: Specific plot events, production facts, experiential feelings

#### Viewer Experience
**Contains**: What it FEELS like to watch — emotional palette, tension/pacing, tone, cognitive complexity, disturbance level, ending aftertaste. **Includes embedded negations** ("no jump scares", "not depressing").

**Good for**: "heartwarming", "slow burn", "campy", "not too dark", "gut-punch ending"

**Not good for**: Plot events, production facts, viewing scenarios

#### Watch Context
**Contains**: WHY and WHEN to watch — motivations, scenarios, feature draws, audience fit.

**Good for**: "date night", "turn my brain off", "great soundtrack", "family friendly"

**Not good for**: Plot events, thematic analysis, technique labels

#### Narrative Techniques
**Contains**: HOW stories are told — POV/perspective, temporal structure, information control, character craft, arc structures, meta techniques.

**Good for**: "unreliable narrator", "nonlinear timeline", "twist ending", "fourth wall breaks"

**Not good for**: Plot content, thematic meaning, production facts, general feelings

#### Production
**Contains**: Pre-release facts — release timing, country of origin, filming location, language, studios, cast/crew, medium, source material, budget scale.

**Good for**: "from the 90s", "French film", "directed by Nolan", "based on true story", "practical effects"

**Not good for**: Story setting ("set in Paris"), aesthetic vibes ("90s vibe"), awards

#### Reception
**Contains**: Post-release evaluation — acclaim level, awards, praised qualities, criticized flaws, audience reactions, controversy.

**Good for**: "Oscar-winning", "critically acclaimed", "great acting", "divisive", "cult classic"

**Not good for**: Neutral plot description, pure production facts, technique labels without evaluation

---

## Vector Weights (`VectorWeightsResponse`)

Determines the relative importance of each vector collection for the given query. This helps the retrieval system allocate search resources appropriately.

### Schema

```python
class VectorWeightsResponse(BaseModel):
    plot_events_data: VectorCollectionWeightData
    plot_analysis_data: VectorCollectionWeightData
    viewer_experience_data: VectorCollectionWeightData
    watch_context_data: VectorCollectionWeightData
    narrative_techniques_data: VectorCollectionWeightData
    production_data: VectorCollectionWeightData
    reception_data: VectorCollectionWeightData

class VectorCollectionWeightData(BaseModel):
    relevance: RelevanceSize
    justification: str  # Brief explanation (10 words or less)
```

### Fields

Each `VectorCollectionWeightData` contains:

| Field | Type | Description |
|-------|------|-------------|
| `relevance` | `RelevanceSize` | How relevant this vector collection is to the query |
| `justification` | `str` | Brief explanation of the relevance assessment |

---

## Shared Enums Reference

### `RelevanceSize` Enum

Used throughout the schema to indicate relative importance:

```python
class RelevanceSize(Enum):
    NOT_RELEVANT = "not_relevant"  # Query has no intent for this channel/vector
    SMALL = "small"                # Minor relevance or ambiguous phrasing
    MEDIUM = "medium"              # Moderate portion of query intent
    LARGE = "large"                # Primary focus of query intent
```

#### Calibration Guidelines

- **`not_relevant`**: The query has absolutely no intent relevant to this channel
- **`small`**: A small portion of the query's intent is relevant, OR there's ambiguous phrasing that COULD be relevant
- **`medium`**: A moderate portion of the query's intent is relevant
- **`large`**: Nearly all of the query's intent is relevant to this channel

**When in doubt** between `small` and `not_relevant`, prefer `small` to avoid missing potentially relevant results.

---

## Complete Example

For the query: **"90s action movies like Die Hard with practical effects, something fun for movie night"**

```json
{
  "channel_weights": {
    "lexical_relevance": "small",
    "metadata_relevance": "medium",
    "vector_relevance": "large"
  },
  "lexical_entities": {
    "entity_candidates": [
      {
        "candidate_entity_phrase": "Die Hard",
        "most_likely_category": "movie_title",
        "exclude_from_results": false,
        "corrected_and_normalized_entity": "Die Hard"
      }
    ]
  },
  "metadata_preferences": {
    "release_date_preference": {
      "result": {
        "first_date": "1990-01-01",
        "match_operation": "between",
        "second_date": "1999-12-31"
      }
    },
    "duration_preference": { "result": null },
    "genres_preference": {
      "result": {
        "should_include": ["Action"],
        "should_exclude": []
      }
    },
    "audio_languages_preference": { "result": null },
    "watch_providers_preference": { "result": null },
    "maturity_rating_preference": { "result": null },
    "popular_trending_preference": {
      "prefers_trending_movies": false,
      "prefers_popular_movies": false
    },
    "reception_preference": {
      "reception_type": "no_preference"
    }
  },
  "vector_subqueries": {
    "plot_events_data": {
      "justification": "Die Hard plot elements",
      "relevant_subquery_text": "hostage situation, terrorists take over building, lone hero fights terrorists, trapped in building"
    },
    "plot_analysis_data": {
      "justification": "action genre, Die Hard themes",
      "relevant_subquery_text": "action thriller, everyman hero, survival against odds, one man army"
    },
    "viewer_experience_data": {
      "justification": "fun, action feel",
      "relevant_subquery_text": "fun, exciting, thrilling, high adrenaline, entertaining, crowd-pleaser"
    },
    "watch_context_data": {
      "justification": "movie night scenario",
      "relevant_subquery_text": "movie night, fun watch, entertaining, practical effects, impressive stunts"
    },
    "narrative_techniques_data": {
      "justification": "no technique content",
      "relevant_subquery_text": null
    },
    "production_data": {
      "justification": "90s, practical effects",
      "relevant_subquery_text": "1990s, 90s, practical effects, practical special effects"
    },
    "reception_data": {
      "justification": "fun = review language",
      "relevant_subquery_text": "fun, entertaining, crowd-pleaser, exciting, thrilling"
    }
  },
  "vector_weights": {
    "plot_events_data": {
      "relevance": "medium",
      "justification": "Die Hard comparison implies similar plot"
    },
    "plot_analysis_data": {
      "relevance": "large",
      "justification": "action genre, thematic similarity"
    },
    "viewer_experience_data": {
      "relevance": "large",
      "justification": "fun is experiential descriptor"
    },
    "watch_context_data": {
      "relevance": "large",
      "justification": "movie night scenario explicit"
    },
    "narrative_techniques_data": {
      "relevance": "not_relevant",
      "justification": "no technique content"
    },
    "production_data": {
      "relevance": "large",
      "justification": "90s decade, practical effects explicit"
    },
    "reception_data": {
      "relevance": "medium",
      "justification": "fun is review language"
    }
  }
}
```

---

## Summary

The `QueryUnderstandingResponse` schema enables our multi-channel retrieval system to:

1. **Route appropriately**: Channel weights determine how much to rely on lexical, metadata, and vector search
2. **Match entities exactly**: Lexical entities enable precise lookups for named things
3. **Filter efficiently**: Metadata preferences enable fast filtering on structured attributes
4. **Search semantically**: Vector subqueries and weights enable nuanced semantic matching across seven specialized embedding spaces

Understanding this schema is essential for debugging search quality issues, extending the query understanding pipeline, and building new features on top of our retrieval system.
