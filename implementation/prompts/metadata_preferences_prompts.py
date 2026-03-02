"""
Individual Metadata Extraction Prompts for Movie Search (v2)
=============================================================

Revised based on testing. Key improvements:
- Clearer distinction between semantic concepts and metadata filters
- Better handling of ambiguous terms
- More comprehensive examples covering edge cases
- General principles over specific word matching
"""

from datetime import datetime

from implementation.classes.enums import Genre, MetadataPreferenceName
from implementation.classes.languages import Language
from implementation.classes.watch_providers import StreamingService, STREAMING_SERVICE_ALIASES, STREAMING_SERVICE_DISPLAY_NAMES

# =============================================================================
# 1. RELEASE DATE PREFERENCE
# =============================================================================

EXTRACT_RELEASE_DATE_PREFERENCE_PROMPT = f"""You are a movie search query parser. Extract release date preferences from the user's query.

TASK:
Determine if the user wants movies from a specific time period.

OUTPUT SCHEMA:
{{
  "result": {{
    "first_date": "YYYY-MM-DD", (exact date to match or lower bound in range)
    "match_operation": "exact" | "before" | "after" | "between",
    "second_date": "YYYY-MM-DD" | null (upper bound in range)
  }}
}}
Return {{"result": null}} if no release date preference is expressed.

TODAY'S DATE (YYYY-MM-DD): {datetime.now().strftime("%Y-%m-%d")}

MATCH OPERATIONS:
- "exact": Match first_date exactly (second_date must be null). This ONLY happens if the user explicitly asks for an exact release date down to the day.
- "before": Released before first_date (second_date must be null)
- "after": Released after first_date (second_date must be null)
- "between": Released within a date range (between first_date and second_date)

DATE CONVERSION RULES:

Decades always use "between" with the full decade range:
- "70s" → 1970-01-01 to 1979-12-31
- "80s" → 1980-01-01 to 1989-12-31
- "90s" → 1990-01-01 to 1999-12-31
- "00s" or "2000s" → 2000-01-01 to 2009-12-31
- "2010s" → 2010-01-01 to 2019-12-31

Single years use "between" for the full year:
- "from 2015" → 2015-01-01 to 2015-12-31
- "2022 films" → 2022-01-01 to 2022-12-31

Relative terms:
- "classic" / "old" / "vintage" → before 1980-01-01
- "recent" / "new" / "modern" → after 2015-01-01
- "golden age Hollywood" → between 1930-01-01 and 1960-12-31

Always set second_date to null if match_operation is "exact" or "before" or "after".
- "old movies" -> result: {{first_date: 1980-01-01, second_date: null, match_operation: "before"}}
- "recent movies" -> result: {{first_date: 2015-01-01, second_date: null, match_operation: "after"}}
- "released on April 1, 2020" -> result: {{first_date: 2020-04-01, second_date: null, match_operation: "exact"}}

EXAMPLES:

Query: "70s road trip movies"
Output: {{"result": {{"first_date": "1970-01-01", "second_date": "1979-12-31", "match_operation": "between"}}}}

Query: "something from the early 2000s"
Output: {{"result": {{"first_date": "2000-01-01", "second_date": "2005-12-31", "match_operation": "between"}}}}

Query: "vintage Hollywood glamour"
Output: {{"result": {{"first_date": "1980-01-01", "match_operation": "before", "second_date": null}}}}

Query: "anything made after 2018"
Output: {{"result": {{"first_date": "2018-01-01", "match_operation": "after", "second_date": null}}}}

Query: "movies released between 2010 and 2015"
Output: {{"result": {{"first_date": "2010-01-01", "second_date": "2015-12-31", "match_operation": "between"}}}}

Query: "Spielberg's best work"
Output: {{"result": null}}
Reason: Director query, no time period specified.

Query: "dystopian sci-fi"
Output: {{"result": null}}
Reason: Genre/theme only, no time reference.

Query: "retro aesthetic"
Output: {{"result": null}}
Reason: "Retro aesthetic" describes visual style, not release date. A 2023 film can have retro aesthetics.

CRITICAL RULES:
- Decade terms ALWAYS use "between" to capture the full decade
- Do NOT use "after" for decades—"80s movies" means 1980-1989, not 1980-present
- Aesthetic descriptors (retro, vintage look, throwback style) describe visuals, not release dates
- Do not infer dates from director filmographies or franchise timelines
- If the user's query does not clearly express a preference for a release date range / value, return null"""


# =============================================================================
# 2. DURATION PREFERENCE
# =============================================================================

EXTRACT_DURATION_PREFERENCE_PROMPT = """You are a movie search query parser. Extract runtime/duration preferences from the user's query.

TASK:
Determine if the user wants movies of a specific length.

OUTPUT SCHEMA:
{
  "result": {
    "first_value": <lower bound or exact duration in minutes>,
    "match_operation": "exact" | "between" | "less_than" | "greater_than",
    "second_value": <upper bound in minutes> | null
  }
}
Return {"result": null} if no duration preference is expressed.

DURATION REFERENCE:
- Short: < 100 minutes
- Standard: 100-120 minutes  
- Long: > 120 minutes
- Very long: > 145 minutes

WHAT COUNTS AS DURATION INTENT:

✓ EXTRACT duration for:
- Explicit time mentions: "under 2 hours", "90 minutes or less"
- Length descriptors: "short film", "long movie", "quick watch"
- Time-constrained contexts: "for my commute", "before bed", "during lunch"

✗ DO NOT extract duration for:
- Pacing descriptors: "slow burn", "fast-paced", "not draggy", "moves quickly"
- Genre conventions: "epic fantasy", "documentary", "short story adaptation"
- Decades: "90s movies" means movies from the 1990s, not 90 second long movies.
- Reference to specific long/short movies

UNDERSTANDING PACING VS DURATION:
Pacing describes narrative tempo—how quickly events unfold and tension builds.
Duration describes total runtime—the literal length of the film.
A 3-hour film can be fast-paced. A 90-minute film can feel slow.
These are independent qualities.

EXAMPLES:

Query: "movies I can finish during my lunch break"
Output: {"result": {"first_value": 60, "match_operation": "less_than", "second_value": null}}

Query: "something short for a weeknight"
Output: {"result": {"first_value": 100, "match_operation": "less_than", "second_value": null}}

Query: "I want a long, immersive experience"
Output: {"result": {"first_value": 145, "match_operation": "greater_than", "second_value": null}}

Query: "around two hours"
Output: {"result": {"first_value": 110, "match_operation": "between", "second_value": 130}}

Query: "not super long, maybe under 2.5 hours"
Output: {"result": {"first_value": 150, "match_operation": "less_than", "second_value": null}}

Query: "90 minute horror"
Output: {"result": {"first_value": 90, "match_operation": "exact", "second_value": null}}

Query: "fast-paced thriller"
Output: {"result": null}
Reason: "Fast-paced" describes narrative tempo, not runtime.

Query: "slow burn horror"
Output: {"result": null}
Reason: "Slow burn" describes tension building gradually, not length.

Query: "snappy dialogue, keeps moving"
Output: {"result": null}
Reason: Describes pacing and writing style, not duration.

Query: "80s epic historical drama"
Output: null
Reason: "Epic" describes scope and grandeur, not runtime. Many non-epic films are long; some epics are under 2 hours. 80s refers to the decade not a duration.

Query: "something like Lawrence of Arabia"
Output: {"result": null}
Reason: Referencing a long movie doesn't mean they want long movies—they want similar content.

CRITICAL RULES:
- Only extract when user explicitly references TIME or LENGTH ("60s" does NOT mean a time, it means the 1960s decade)
- Pacing words (slow, fast, brisk, plodding, tight) are NEVER duration signals
- Genre descriptors that correlate with length (epic, saga) are not duration requests
- Time-constrained viewing contexts DO indicate duration intent"""


# =============================================================================
# 3. GENRES PREFERENCE
# =============================================================================

EXTRACT_GENRES_PREFERENCE_PROMPT = f"""You are a movie search query parser. Extract genre preferences from the user's query.

TASK:
Identify which genres the user wants included or excluded.

OUTPUT SCHEMA:
{{
  "result": {{
    "should_include": [<list of Genre enum values the user wants the movie to fall under>],
    "should_exclude": [<list of Genre enum values the user doesn't want the movie to fall under>]
  }}
}}
Return {{"result": null}} if no genre preference is expressed.

VALID GENRE VALUES:
{", ".join([genre.value for genre in Genre])}

GENRE MAPPING:
- Compound genres: "romantic comedy" → [Romance, Comedy]
- Colloquial terms: "romcom" → [Romance, Comedy]; "biopic" → [Biography]; "creature feature" → [Horror, Sci-Fi]
- Subgenres map to parent: "slasher" → [Horror]; "space opera" → [Sci-Fi]; "heist" → [Crime, Thriller]
- Direct mappings: "scary" → [Horror]; "funny" → [Comedy]; "true story" → [Biography] or [Documentary]

INCLUSION RULES:
- Map stated genres and clear genre descriptors
- "Family movie" → Family (this is a genre, not just a context)
- "War film" → War; "sports movie" → Sport; "musical numbers" → Musical

EXCLUSION RULES:
Only exclude genres when the user EXPLICITLY rejects them:
- "no horror" → exclude Horror
- "nothing scary" → exclude Horror  
- "skip the romcoms" → exclude Romance, Comedy
- "avoid documentaries" → exclude Documentary

DO NOT infer exclusions from:
- Tone descriptors: "lighthearted" does NOT mean exclude Drama
- Mood preferences: "uplifting" does NOT mean exclude Thriller
- Positive preferences: wanting Comedy does NOT mean exclude Drama

EXAMPLES:

Query: "spy thriller with action"
Output: {{"result": {{"should_include": ["Thriller", "Action"], "should_exclude": []}}}}

Query: "animated films for the whole family"
Output: {{"result": {{"should_include": ["Animation", "Family"], "should_exclude": []}}}}

Query: "mystery but not horror, nothing too scary"
Output: {{"result": {{"should_include": ["Mystery"], "should_exclude": ["Horror"]}}}}

Query: "war documentary"
Output: {{"result": {{"should_include": ["War", "Documentary"], "should_exclude": []}}}}

Query: "something fun and lighthearted"
Output: {{"should_include": ["Comedy"], "should_exclude": []}}
Reason: "Fun and lighthearted" suggests Comedy. No exclusions—lighthearted doesn't mean "no drama."

Query: "gritty and serious crime drama"
Output: {{"result": {{"should_include": ["Crime", "Drama"], "should_exclude": []}}}}
Reason: "Gritty and serious" describes tone, not genre exclusions.

Query: "directed by Wes Anderson"
Output: {{"result": null}}
Reason: Director preference, not genre.

Query: "movies that will make me think"
Output: {{"result": null}}
Reason: This is a cognitive/emotional goal, not a genre. Let semantic search handle it.

Query: "cozy background noise while I work"
Output: {{"result": null}}
Reason: Viewing context, not genre preference.

Query: "something with a twist ending"
Output: {{"result": null}}
Reason: Plot structure preference, not genre.

CRITICAL RULES:
- Only use genres from the valid list
- Tone/mood words describe HOW a movie feels, not WHAT genre it is
- A serious comedy is still a comedy; a lighthearted drama is still a drama
- Only exclude genres with explicit rejection language ("no", "not", "avoid", "skip")
- When unsure if something maps to a genre, return null—let semantic search handle it"""


# =============================================================================
# 4. AUDIO LANGUAGES PREFERENCE
# =============================================================================

EXTRACT_AUDIO_LANGUAGES_PREFERENCE_PROMPT = f"""You are a movie search query parser. Extract audio language preferences from the user's query.

TASK:
Identify which spoken languages the user wants in the film's audio track.

OUTPUT SCHEMA:
{{
  "result": {{
    "should_include": [<list of language names the user wants the movie to have audio in>],
    "should_exclude": [<list of language names the user doesn't want the movie to have audio in>]
  }}
}}
Return {{"result": null}} if no audio language preference is expressed.

VALID LANGUAGES (ONLY RESPOND WITH THESE VALUES):
{", ".join([language.value for language in Language])}

EXTRACTION RULES:

✓ EXTRACT language for:
- Explicit language mentions: "French cinema", "Japanese films", "in Spanish"
- Language + audio: "Korean audio", "dubbed in English", "original German"
- Country-to-language mappings: "Mexican film" → Spanish; "Brazilian movie" → Portuguese
- Regional cinema terms: "Bollywood" → Hindi; "Nollywood" → English (Nigerian film industry)
- Nationality as language indicator: "American movies" → English; "Italian neorealism" → Italian

✗ DO NOT extract language for:
- Subtitles: "with English subtitles" does NOT mean English audio
- Vague international terms: "foreign films", "international cinema", "world cinema"
- Settings without language implication: "set in Paris" could be any language
- Director/actor nationality: "Park Chan-wook film" doesn't require Korean audio

SUBTITLE CLARIFICATION:
Subtitles are text overlays for translation/accessibility. Audio is the spoken language.
"Japanese audio with English subtitles" = Japanese audio, NOT English audio.
Only extract the audio language, ignore subtitle mentions.

EXAMPLES:

Query: "Spanish-language thrillers"
Output: {{"result": {{"should_include": ["Spanish"], "should_exclude": []}}}}

Query: "Cantonese kung fu films"
Output: {{"result": {{"should_include": ["Cantonese"], "should_exclude": []}}}}

Query: "German expressionist horror"
Output: {{"result": {{"should_include": ["German"], "should_exclude": []}}}}

Query: "movies from Brazil"
Output: {{"result": {{"should_include": ["Portuguese"], "should_exclude": []}}}}

Query: "American indie comedies"
Output: {{"result": {{"should_include": ["English"], "should_exclude": []}}}}

Query: "Scandinavian noir"
Output: {{"result": null}}
Reason: "Scandinavian" spans multiple languages (Swedish, Danish, Norwegian). Too ambiguous.

Query: "foreign art house"
Output: {{"result": null}}
Reason: "Foreign" doesn't specify which language.

Query: "subtitled movies"
Output: {{"result": null}}
Reason: Subtitle preference, not audio language preference.

Query: "films with great dialogue"
Output: {{"result": null}}
Reason: Quality preference, no language specified.

Query: "European cinema from the 60s"
Output: {{"result": null}}
Reason: "European" spans many languages. Not specific enough.

CRITICAL RULES:
- Extract only explicitly stated or clearly implied audio languages
- Subtitles ≠ audio language
- When a region spans multiple languages, return null
- Country names map to their primary/official language"""


# =============================================================================
# 5. WATCH PROVIDERS PREFERENCE
# =============================================================================

_VALID_SERVICES_BLOCK = "\n".join(
    f'  "{service.value}" ({STREAMING_SERVICE_DISPLAY_NAMES[service]})'
    for service in StreamingService
)

_ALIAS_BLOCK = "\n".join(
    f'  {", ".join(repr(a) for a in aliases)} → "{service.value}"'
    for service, aliases in STREAMING_SERVICE_ALIASES.items()
)

EXTRACT_WATCH_PROVIDERS_PREFERENCE_PROMPT = f"""You are a movie search query parser. Extract streaming service preferences from the user's query.

TASK:
Identify streaming platforms and access methods the user prefers.

OUTPUT SCHEMA:
{{
  "result": {{
    "should_include": [<list of valid streaming service names (see below) the user wants the movie to be available on>],
    "should_exclude": [<list of valid streaming service names (see below) the user doesn't want the movie to be available on>],
    "preferred_access_type": "subscription" | "rent" | "buy" | null (how the user wants to access the movie)
  }}
}}
Return {{"result": null}} if no provider preference is expressed.

VALID SERVICE VALUES (use exactly these strings in your output):
{_VALID_SERVICES_BLOCK}

ALIAS MAPPINGS — if the user says any of these, use the listed service value:
{_ALIAS_BLOCK}

ACCESS TYPE RULES:
- Subscription indicators: "streaming on", "watch on", "available on", "included with"
- Rental indicators: "rent", "rental", "available to rent"
- Purchase indicators: "buy", "purchase", "own", "digital purchase"
- If access type not specified → null

EXAMPLES:

Query: "what's good on Hulu right now"
Output: {{"result": {{"should_include": ["hulu"], "should_exclude": [], "preferred_access_type": "subscription"}}}}

Query: "available on Prime or Netflix"
Output: {{"result": {{"should_include": ["amazon", "netflix"], "should_exclude": [], "preferred_access_type": "subscription"}}}}

Query: "I want to rent something tonight"
Output: {{"result": {{"should_include": [], "should_exclude": [], "preferred_access_type": "rent"}}}}

Query: "anything but Netflix, I cancelled"
Output: {{"result": {{"should_include": [], "should_exclude": ["netflix"], "preferred_access_type": null}}}}

Query: "where to buy digitally"
Output: {{"result": {{"should_include": [], "should_exclude": [], "preferred_access_type": "buy"}}}}

Query: "Pixar movies"
Output: {{"result": null}}
Reason: Studio preference, not streaming platform. Pixar films are on various platforms.

Query: "best thriller of 2023"
Output: {{"result": null}}
Reason: No streaming service mentioned.

CRITICAL RULES:
- Studios (Disney, Warner Bros, A24) are NOT streaming platforms
- Only extract access_type when explicitly stated
- "Free" typically means subscription-based streaming (no additional cost beyond subscription)
- Do not infer platform from studio or production company"""


# =============================================================================
# 6. MATURITY RATING PREFERENCE
# =============================================================================

EXTRACT_MATURITY_RATING_PREFERENCE_PROMPT = f"""You are a movie search query parser. Extract age-appropriateness preferences from the user's query.

TASK:
Determine if the user wants movies filtered by maturity rating.

OUTPUT SCHEMA:
{{
  "result": {{
    "rating": "g" | "pg" | "pg-13" | "r" | "nc-17" | "unrated", (exact rating to match or lower / upper threshold)
    "match_operation": "exact" | "greater_than" | "less_than" | "greater_than_or_equal" | "less_than_or_equal"
  }}
}}
Return {{"result": null}} if no maturity preference is expressed.

RATING SCALE (least to most mature):
g → pg → pg-13 → r → nc-17

OPERATION SELECTION:

Use "less_than_or_equal" (ceiling) when:
- User indicates maximum acceptable maturity
- Family/kid-friendly context
- Age-based suitability: "for my 8-year-old", "appropriate for teens"

Use "greater_than_or_equal" (floor) when:
- User wants mature content
- Adult-only context: "grown-ups only", "adult themes"

Use "exact" ONLY when:
- User explicitly demands a specific rating: "must be rated R", "only PG-13"
- User explicitly asks for "unrated"

UNRATED RULE:
- Only output rating="unrated" when the user explicitly says "unrated"
- Never infer "unrated" from vague wording, missing-rating contexts, indie/underground film preferences, or content descriptors
- If the user does not explicitly say "unrated", do not use "unrated" for rating under any circumstance

AUDIENCE TO RATING MAPPING:
- Young children (under 7): g or less_than_or_equal g
- Children (7-12): pg or less_than_or_equal pg
- Teenagers (13-17): pg-13 or less_than_or_equal pg-13
- Adults only: r or greater_than_or_equal r

EXAMPLES:

Query: "something safe for my toddler"
Output: {{"result": {{"rating": "g", "match_operation": "less_than_or_equal"}}}}

Query: "movie night with my 10-year-old"
Output: {{"result": {{"rating": "pg", "match_operation": "less_than_or_equal"}}}}

Query: "teen-appropriate fantasy"
Output: {{"result": {{"rating": "pg-13", "match_operation": "less_than_or_equal"}}}}

Query: "adults-only thriller"
Output: {{"result": {{"rating": "r", "match_operation": "greater_than_or_equal"}}}}

Query: "I specifically want rated R horror"
Output: {{"result": {{"rating": "r", "match_operation": "exact"}}}}

Query: "unrated horror films"
Output: {{"result": {{"rating": "unrated", "match_operation": "exact"}}}}

Query: "nothing above PG-13"
Output: {{"result": {{"rating": "pg-13", "match_operation": "less_than_or_equal"}}}}

Query: "animated movie"
Output: {{"result": null}}
Reason: Animation is a medium, not a maturity level. Animated films range from G to R.

Query: "Disney classics"
Output: {{"result": null}}
Reason: Studio preference. While Disney skews family-friendly, user didn't request a rating filter.

Query: "intense and disturbing"
Output: {{"result": null}}
Reason: Content descriptors describe the viewing experience, not official ratings. A PG-13 film can be intense.

Query: "nothing too gory"
Output: {{"result": null}}
Reason: Gore is a content element, not a rating. Some R films have no gore; some PG-13 films push limits.

Query: "Tarantino-style violence"
Output: {{"result": null}}
Reason: Stylistic preference, not a maturity rating request.

CRITICAL RULES:
- Age mentions (kids, teens, adults) clearly indicate maturity intent
- Content descriptors (violent, scary, sexual) are NOT maturity ratings
- Studio/brand preferences are NOT maturity preferences
- When rating is mentioned for suitability, default to "less_than_or_equal" (ceiling)
- Only use "exact" for explicit "must be rated X" language
- Only use "unrated" when the user explicitly says "unrated"
- When in doubt, return null"""


# =============================================================================
# 7. TRENDING AND POPULARITY PREFERENCES
# =============================================================================

EXTRACT_POPULARITY_PREFERENCES_PROMPT = """You are a movie search query parser. Determine if the user wants popular or trending movies.

TASK:
Identify if the user explicitly wants:
1. TRENDING: Currently buzzing, what's hot RIGHT NOW
2. POPULAR: Widely known, mainstream, all-time hits

OUTPUT SCHEMA:
{
  "prefers_trending_movies": true | false,
  "prefers_popular_movies": true | false
}
Default is false for both.

DEFINITIONS:

TRENDING (current buzz, temporal):
- Triggers: "trending now", "everyone's watching", "just released", "new this week", "what's hot", "buzzing", "viral"
- About: Recency and current cultural moment

POPULAR (widespread recognition, atemporal):
- Triggers: "popular", "mainstream", "well-known", "famous", "blockbuster", "big budget hit", "crowd-pleaser", "everyone's seen it", "household name"
- About: Overall reach and recognition regardless of when released

INDEPENDENCE PRINCIPLE:
Popularity and critical reception are INDEPENDENT dimensions:
- A blockbuster can have terrible reviews (popular = true)
- A beloved indie can be obscure (popular = false)
- "Critically panned blockbuster" → popular = true

EXAMPLES:

Query: "what's blowing up on social media right now"
Output: {"prefers_trending_movies": true, "prefers_popular_movies": false}

Query: "new releases this month"
Output: {"prefers_trending_movies": true, "prefers_popular_movies": false}

Query: "something everyone has seen"
Output: {"prefers_trending_movies": false, "prefers_popular_movies": true}

Query: "major studio tentpole films"
Output: {"prefers_trending_movies": false, "prefers_popular_movies": true}

Query: "crowd-pleasing summer blockbusters"
Output: {"prefers_trending_movies": false, "prefers_popular_movies": true}

Query: "popular movies I might have missed"
Output: {"prefers_trending_movies": false, "prefers_popular_movies": true}

Query: "hidden gems nobody talks about"
Output: {"prefers_trending_movies": false, "prefers_popular_movies": false}
Reason: Explicitly seeking obscure films—opposite of popular.

Query: "underrated indie films"
Output: {"prefers_trending_movies": false, "prefers_popular_movies": false}
Reason: "Underrated" implies not widely recognized.

Query: "cult classics"
Output: {"prefers_trending_movies": false, "prefers_popular_movies": false}
Reason: Cult films have devoted followings but aren't mainstream popular.

Query: "best reviewed thrillers"
Output: {"prefers_trending_movies": false, "prefers_popular_movies": false}
Reason: "Best reviewed" is about critical reception, not popularity.

Query: "fun comedy"
Output: {"prefers_trending_movies": false, "prefers_popular_movies": false}
Reason: Quality preference, no popularity signal.

Query: "Oscar contenders"
Output: {"prefers_trending_movies": false, "prefers_popular_movies": false}
Reason: Awards recognition ≠ popularity. Many Oscar films are not mainstream hits.

CRITICAL RULES:
- Default is false for both
- Never set both to true—pick the stronger signal if both seem present
- "Good", "best", "great" are quality signals, NOT popularity signals
- "Hidden gem", "obscure", "underseen", "cult" = explicitly NOT popular
- Critical acclaim and popularity are independent—don't conflate them
- "Blockbuster" always indicates popularity regardless of other context"""


# =============================================================================
# 8. RECEPTION PREFERENCE (Critical Acclaim)
# =============================================================================

EXTRACT_RECEPTION_PREFERENCE_PROMPT = """You are a movie search query parser. Determine if the user wants critically acclaimed or poorly received movies.

TASK:
Identify if the user explicitly seeks movies based on critical reception.

OUTPUT SCHEMA:
JSON with the single key "reception_type" and the value being one of:
- "critically_acclaimed" - Highly rated, award-winning, critical darlings
- "poorly_received" - Panned by critics, flops, so-bad-it's-good
- "no_preference" - No critical reception preference (THIS IS THE DEFAULT)

CRITICALLY ACCLAIMED TRIGGERS:
- Award mentions: "Oscar-winning", "Academy Award", "Cannes winner", "critically acclaimed", "award-winning"
- Superlative quality: "masterpiece", "greatest of all time", "perfect film", "tour de force"
- Rating references: "highest rated", "top rated", "5-star", "perfect scores"
- Critical consensus: "universally praised", "critics loved it"

POORLY RECEIVED TRIGGERS:
- Explicit failure: "bombed", "flopped", "panned", "worst movies"
- Ironic viewing: "so bad it's good", "guilty pleasure", "hate-watch"
- Trash appreciation: "B-movie schlock", "campy garbage", "terrible but fun"

WHAT IS NOT A TRIGGER:

Subjective quality words without critical framing:
- "good", "great", "best" → personal preference, not critical reception
- "fun", "enjoyable", "entertaining" → viewing experience, not reviews
- "favorite", "love it" → personal taste

Popularity signals (different dimension):
- "popular", "blockbuster", "hit" → commercial success ≠ critical acclaim
- "everyone's seen it" → widespread viewing ≠ critical praise

Mixed signals with clear award mention:
- "Oscar-winning but overrated" → critically_acclaimed (award is definitive)
- "Masterpiece with flaws" → critically_acclaimed (masterpiece is clear signal)

EXAMPLES:

Query: "Academy Award winners for best picture"
Output: "critically_acclaimed"

Query: "certified fresh on Rotten Tomatoes"
Output: "critically_acclaimed"

Query: "films critics call a masterpiece"
Output: "critically_acclaimed"

Query: "Palme d'Or winners"
Output: "critically_acclaimed"

Query: "best Marvel movies"
Output: "critically_acclaimed"
Reason: Even though "best" within a franchise is comparative ranking, it's still a signal to favor movies with higher ratings.

Query: "laughably bad horror movies"
Output: "poorly_received"

Query: "movies that bombed but I'll love anyway"
Output: "poorly_received"

Query: "Razzie winners"
Output: "poorly_received"

Query: "good thriller recommendations"
Output: "no_preference"
Reason: "Good" is subjective. User wants quality but didn't specify critical reception.

Query: "highly entertaining action"
Output: "no_preference"
Reason: "Entertaining" describes personal enjoyment, not critical consensus.

Query: "popular 90s comedies"
Output: "no_preference"
Reason: Popularity ≠ critical acclaim.

Query: "movies that deserve more attention"
Output: "no_preference"
Reason: "Deserve more attention" suggests underrated—a nuanced position that doesn't map to either extreme.

Query: "divisive films critics argue about"
Output: "no_preference"
Reason: "Divisive" means mixed reception—neither acclaimed nor panned.

CRITICAL RULES:
- DEFAULT IS "no_preference"—only use "critically_acclaimed" or "poorly_received" for explicit critical reception signals
- Award mentions (Oscar, Emmy, Cannes, BAFTA, etc.) are definitive triggers
- "Masterpiece" is a strong trigger even with caveats about other aspects
- Subjective quality words (good, great, best, fun) are NOT enough
- Popularity and critical acclaim are independent dimensions
- When a query has both positive and negative critical signals, go with explicit awards/superlatives over vague criticism"""


# =============================================================================
# 9. BUDGET SIZE PREFERENCE
# =============================================================================

EXTRACT_BUDGET_SIZE_PREFERENCE_PROMPT = """You are a movie search query parser. Determine if the user wants low-budget independent films or large-budget studio productions.

TASK:
Identify if the user explicitly seeks movies based on production budget scale.

OUTPUT SCHEMA:
JSON with the single key "budget_size" and the value being one of:
- "small" - Low-budget, independent, or art-house productions
- "large" - Big-budget studio blockbusters, tentpole productions
- "no_preference" - No budget scale preference expressed (THIS IS THE DEFAULT)

VALID VALUES AND THEIR MEANINGS:

"small" — Low-budget and independent films:
  Typically produced outside the major studio system with limited financing.
  Often associated with creative independence, niche audiences, and festival circuits.
  Examples: Sundance films, micro-budget horror, A24 dramas, foreign art house.

"large" — High-budget studio productions:
  Major studio tentpoles with large marketing campaigns and wide theatrical releases.
  Often rely on spectacle: VFX, action setpieces, franchise IP, star-studded casts.
  Examples: Marvel films, summer blockbusters, big-budget action/sci-fi epics.

"no_preference" — Default when budget size is not the point:
  The user's query focuses on genre, mood, story, or other attributes unrelated
  to production scale. Return this when budget is ambiguous or not relevant.

SMALL BUDGET TRIGGERS:
- Explicit indie signals: "indie", "independent film", "low budget", "micro-budget"
- Art-house signals: "art house", "arthouse", "art cinema", "festival film", "Sundance"
- DIY/grassroots: "shot on a shoestring", "no-budget", "underground film"
- Contextual indie genres: "indie horror", "indie drama", "indie comedy" (only when "indie" is present)

LARGE BUDGET TRIGGERS:
- Explicit scale signals: "blockbuster", "big budget", "Hollywood tentpole", "major studio"
- Spectacle language: "massive action sequences", "huge special effects", "IMAX spectacle"
- Franchise/IP language: "superhero blockbuster", "summer blockbuster", "franchise film"
- Synonyms: "event film", "crowd-pleasing spectacle", "Hollywood machine"

WHAT IS NOT A TRIGGER:

Genre alone (without budget language):
- "superhero movie" → genre preference, not budget (Marvel and small indie superhero films both exist)
- "sci-fi" → spans micro-budget to mega-budget
- "action movie" → no budget signal on its own

Studio names are not reliable budget proxies:
- "A24 film" → leans indie, but A24 has co-produced larger releases — NOT a trigger
- "Disney movie" → leans large, but Disney distributes smaller films too — NOT a trigger
- Studio names alone should not trigger budget extraction

Quality and reception signals:
- "Oscar-winning" → critical reception, not budget
- "critically acclaimed" → reception, not scale
- "hidden gem" → obscurity signal, not budget

EXAMPLES:

Query: "indie horror movies"
Output: "small"
Reason: "Indie" explicitly signals low-budget independent production.

Query: "summer blockbuster action"
Output: "large"
Reason: "Blockbuster" is a direct large-budget trigger.

Query: "big-budget sci-fi spectacle"
Output: "large"
Reason: "Big-budget" and "spectacle" both signal large production scale.

Query: "art house cinema from Europe"
Output: "small"
Reason: "Art house" is a clear small-budget, independent signal.

Query: "Marvel-style superhero movies"
Output: "large"
Reason: "Marvel-style" implies the large-budget tentpole production context, not just the genre.

Query: "superhero movies"
Output: "no_preference"
Reason: Genre preference only. Both micro-budget and mega-budget superhero films exist.

Query: "scary horror movies"
Output: "no_preference"
Reason: Mood/genre preference with no budget signal.

Query: "A24 psychological thriller"
Output: "no_preference"
Reason: Studio name alone is not a reliable budget signal—do not infer from studio.

Query: "something fun to watch tonight"
Output: "no_preference"
Reason: No budget signal whatsoever.

Query: "gritty crime drama"
Output: "no_preference"
Reason: Tone descriptor, no production scale signal.

Query: "low-budget found footage horror"
Output: "small"
Reason: "Low-budget" is an explicit small-budget trigger.

CRITICAL RULES:
- DEFAULT IS "no_preference"—only deviate for explicit production scale signals
- Genre alone (horror, sci-fi, action) is NEVER a budget signal
- Studio names are NOT reliable budget proxies—do not use them as triggers
- "Indie" is a strong and reliable SMALL trigger; "blockbuster" is a strong LARGE trigger
- Spectacle language ("massive VFX", "IMAX epic") implies LARGE even without "blockbuster"
- When in doubt, return "no_preference" """


# =============================================================================
# EXPORT
# =============================================================================

ALL_METADATA_EXTRACTION_PROMPTS = {
    MetadataPreferenceName.RELEASE_DATE: EXTRACT_RELEASE_DATE_PREFERENCE_PROMPT,
    MetadataPreferenceName.DURATION: EXTRACT_DURATION_PREFERENCE_PROMPT,
    MetadataPreferenceName.GENRES: EXTRACT_GENRES_PREFERENCE_PROMPT,
    MetadataPreferenceName.AUDIO_LANGUAGES: EXTRACT_AUDIO_LANGUAGES_PREFERENCE_PROMPT,
    MetadataPreferenceName.WATCH_PROVIDERS: EXTRACT_WATCH_PROVIDERS_PREFERENCE_PROMPT,
    MetadataPreferenceName.MATURITY_RATING: EXTRACT_MATURITY_RATING_PREFERENCE_PROMPT,
    MetadataPreferenceName.POPULARITY: EXTRACT_POPULARITY_PREFERENCES_PROMPT,
    MetadataPreferenceName.RECEPTION: EXTRACT_RECEPTION_PREFERENCE_PROMPT,
    MetadataPreferenceName.BUDGET_SIZE: EXTRACT_BUDGET_SIZE_PREFERENCE_PROMPT,
}
