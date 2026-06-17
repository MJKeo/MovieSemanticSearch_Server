"""
Trait category taxonomy for the search v2 step-2 grounding LLM.

`CategoryName` is the canonical vocabulary the step-2 pre-pass picks
from when grounding each captured meaning of a query fragment. Each
member carries:

- A descriptive name (the string value emitted to downstream code).
- A compact `description` of what the category covers.
- A `boundary` — what the category does NOT own and the explicit
  redirects to the categories that do.
- `edge_cases` — concrete misroute traps with the disambiguator.
- `good_examples` — short trait surface forms that clearly belong.
- `bad_examples` — surface forms that look like they belong but
  route elsewhere, with the redirect spelled out.
- An ordered tuple of `EndpointRoute` values.
- A `HandlerBucket` identifying the query-generation instruction shape.

All five text fields are programmatically inserted into the step-2
system prompt — wording is kept tight to avoid prompt bloat.

The 43-category list and granularity principles live in
`search_improvement_planning/query_categories.md` and
`search_improvement_planning/v3_category_attributes.md`.
"""

from enum import Enum

from schemas.enums import CategoryCombineType, EndpointRoute, HandlerBucket


class CategoryName(str, Enum):
    description: str
    boundary: str
    edge_cases: tuple[str, ...]
    good_examples: tuple[str, ...]
    bad_examples: tuple[str, ...]
    endpoints: tuple["EndpointRoute", ...]
    bucket: "HandlerBucket"
    combine_type: "CategoryCombineType"

    def __new__(
        cls,
        value: str,
        description: str,
        boundary: str,
        edge_cases: tuple[str, ...],
        good_examples: tuple[str, ...],
        bad_examples: tuple[str, ...],
        endpoints: tuple["EndpointRoute", ...],
        bucket: "HandlerBucket",
        combine_type: "CategoryCombineType",
    ) -> "CategoryName":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        obj.boundary = boundary
        obj.edge_cases = edge_cases
        obj.good_examples = good_examples
        obj.bad_examples = bad_examples
        obj.endpoints = endpoints
        obj.bucket = bucket
        obj.combine_type = combine_type
        return obj

    # -----------------------------------------------------------------
    # Structured / lexical
    # -----------------------------------------------------------------

    PERSON_CREDIT = (
        "Person credit",
        "Indexed film credits: actor, director, writer, producer, composer.",
        (
            "Indexed roles only. Cinematographer/editor/PD/costume/VFX "
            "→ BELOW_THE_LINE_CREATOR. Title strings → TITLE_TEXT. "
            "Source authors → NAMED_SOURCE_CREATOR."
        ),
        (
            "Composer is indexed (John Williams score → here, not MUSIC_SCORE_ACCLAIM).",
            "Role markers ('starring', 'directed by') absorb into trait.",
        ),
        ("starring Tom Hanks", "directed by Greta Gerwig", "Hans Zimmer score"),
        (
            "'Roger Deakins-shot' → BELOW_THE_LINE_CREATOR (DP not indexed).",
            "'Stephen King movies' → NAMED_SOURCE_CREATOR (novelist).",
        ),
        (EndpointRoute.ENTITY,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    TITLE_TEXT = (
        "Title text lookup",
        "Substring matches against the movie's title.",
        (
            "Title-string framing only. Person names → PERSON_CREDIT. "
            "Franchise/character names route to their own category "
            "(FRANCHISE_LINEAGE / CHARACTER_FRANCHISE / NAMED_CHARACTER) "
            "even when the name overlaps with title text."
        ),
        (
            "Disambiguator is the framing: 'in the title' / 'called' / "
            "'titled' fires here. Bare names default to entity routing.",
        ),
        ("any movie called Inception", "movies with 'Star' in the title", "titles starting with The"),
        (
            "'Star Wars' → FRANCHISE_LINEAGE (franchise framing).",
            "'Batman' → CHARACTER_FRANCHISE.",
            "'Tom Hanks' → PERSON_CREDIT.",
        ),
        (EndpointRoute.ENTITY,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    NAMED_CHARACTER = (
        "Named character",
        (
            "Specific named persona appearing in films but NOT anchoring a "
            "film franchise. The character recurs across films (or appears "
            "in a notable single film) but is not the binding identity of "
            "a series. The trait must name the specific character; do not "
            "route here when the trait names a character type or archetype."
        ),
        (
            "Non-anchoring characters only. Character-anchored franchises "
            "(Batman, Bond, Spider-Man, Sherlock Holmes, Harry Potter) → "
            "CHARACTER_FRANCHISE. Static character TYPES (anti-hero, femme "
            "fatale) → CHARACTER_ARCHETYPE. Actors → PERSON_CREDIT."
        ),
        (
            "Anchoring test: does a Wikipedia 'List of [X] films' entity "
            "exist with this character as through-line? Bond/Batman/Spider-Man "
            "→ yes → CHARACTER_FRANCHISE. Yoda/Hermione/Loki → no → here.",
            "'Yoda in Star Wars' splits: 'Yoda' here + 'Star Wars' → FRANCHISE_LINEAGE.",
        ),
        ("movies featuring Yoda", "Hermione Granger scenes", "Loki appearances"),
        (
            "'Batman' → CHARACTER_FRANCHISE (anchors a franchise).",
            "'anti-hero protagonist' → CHARACTER_ARCHETYPE (type, not name).",
            "'Daniel Radcliffe' → PERSON_CREDIT (actor not character).",
        ),
        (EndpointRoute.ENTITY,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    STUDIO_BRAND = (
        "Studio / brand",
        (
            "Production studios and curated brands — entities that produce, "
            "finance, or distribute films under a recognizable brand identity. "
            "Covers major studios, mini-majors, prestige labels, and specialty "
            "production houses. The trait must name the specific studio; do "
            "not route here when the trait names a kind of film a studio "
            "happens to produce."
        ),
        (
            "Entity that made the movie. Franchises a studio owns → "
            "FRANCHISE_LINEAGE. Cultural traditions (Bollywood, Korean cinema) "
            "→ CULTURAL_TRADITION."
        ),
        (
            "Some names are both studio and franchise (Marvel Studios vs MCU; "
            "Disney studio vs Disney character canon). Pick the reading that "
            "best fits the surrounding query — no universal default.",
            "Pixar is its own brand even though Disney owns it.",
        ),
        ("A24 horror", "Studio Ghibli movies", "Blumhouse"),
        (
            "'Marvel Cinematic Universe' → FRANCHISE_LINEAGE.",
            "'Bollywood' → CULTURAL_TRADITION.",
        ),
        (EndpointRoute.STUDIO,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    FRANCHISE_LINEAGE = (
        "Franchise / universe lineage",
        (
            "Two things: membership in a character-less film franchise — "
            "a recurring series whose identity is defined by setting, "
            "ensemble, or shared universe rather than any single anchoring "
            "character — AND lineage positioning (sequel/prequel/spinoff/"
            "reboot, mainline vs offshoot, original vs remake). The trait "
            "must name the specific franchise; do not route here when the "
            "trait names a concept that franchises happen to exemplify."
        ),
        (
            "Lineage POSITIONING fires here regardless of franchise type — "
            "'Batman spinoffs' splits 'spinoffs' to here even though 'Batman' "
            "→ CHARACTER_FRANCHISE. Character-anchored franchise NAMES (Batman, "
            "Bond, Spider-Man, Sherlock Holmes, Harry Potter) → CHARACTER_FRANCHISE. "
            "Generic medium named alongside a character-less franchise splits "
            "('Star Wars novelizations' → 'Star Wars' here + 'novelizations' → "
            "ADAPTATION_SOURCE). Source-material creators (Stephen King, "
            "Shakespeare) → NAMED_SOURCE_CREATOR."
        ),
        (
            "'The original, not the remake' is lineage positioning here.",
            "MCU/Star Wars subgroups ('phase 4', 'the prequels') route here.",
        ),
        ("Marvel Cinematic Universe movies", "the Star Wars prequels", "Fast & Furious sequels"),
        (
            "'Batman' → CHARACTER_FRANCHISE.",
            "'based on a comic book' → ADAPTATION_SOURCE (no named franchise).",
        ),
        (EndpointRoute.FRANCHISE_STRUCTURE,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    CHARACTER_FRANCHISE = (
        "Character-franchise",
        (
            "A named character whose identity anchors a film franchise — "
            "the same character appears across multiple installments and "
            "binds the films together such that the franchise is "
            "recognizable by the character's name. The trait must name "
            "the specific anchoring character; do not route here when "
            "the trait names a character type or a kind-of-protagonist "
            "without naming a specific one."
        ),
        (
            "Names the anchoring character only. Sequel/prequel/spinoff/reboot "
            "is a SEPARATE trait → FRANCHISE_LINEAGE. Non-anchoring characters "
            "(supporting roles or universe inhabitants who do not bind their "
            "own franchise) → NAMED_CHARACTER. Character-less franchises "
            "defined by setting or ensemble → FRANCHISE_LINEAGE."
        ),
        (
            "Anchoring test: a 'List of [X] films' Wikipedia entity exists "
            "with this character as the through-line.",
            "'Sherlock Holmes books' splits: 'Sherlock Holmes' here + 'books' "
            "→ ADAPTATION_SOURCE.",
            "Lineage modifiers ('spinoffs', 'reboots') split off the "
            "character name and route to FRANCHISE_LINEAGE.",
        ),
        ("James Bond films", "Indiana Jones movies", "Sherlock Holmes adaptations"),
        (
            "'Yoda appearances' → NAMED_CHARACTER.",
            "'Star Wars movies' → FRANCHISE_LINEAGE.",
            "'Daniel Craig as Bond' splits: 'Daniel Craig' → PERSON_CREDIT + "
            "'Bond' here.",
        ),
        (EndpointRoute.ENTITY, EndpointRoute.FRANCHISE_STRUCTURE),
        HandlerBucket.CHARACTER_FRANCHISE_FANOUT,
        CategoryCombineType.ALTERNATIVES,
    )
    ADAPTATION_SOURCE = (
        "Adaptation source flag",
        (
            "Origin medium of source material — novel, comic book, true "
            "story, biography, video-game adaptation, remake (as origin "
            "medium, not lineage positioning)."
        ),
        (
            "About WHAT the source medium is. When the query also names a "
            "specific source, the named source becomes a SEPARATE trait — "
            "character/franchise → CHARACTER_FRANCHISE / FRANCHISE_LINEAGE, "
            "creator → NAMED_SOURCE_CREATOR. Lineage positioning (original "
            "vs remake) stays in FRANCHISE_LINEAGE."
        ),
        (
            "'Stephen King novels' splits: 'Stephen King' → NAMED_SOURCE_CREATOR "
            "+ 'novels' here.",
            "'Sherlock Holmes books' splits: 'Sherlock Holmes' (book's subject) "
            "→ CHARACTER_FRANCHISE + 'books' here.",
            "'JFK biopic' splits: 'JFK' → CENTRAL_TOPIC + 'biopic' here.",
        ),
        ("novel adaptation", "based on a comic book", "based on a true story"),
        (
            "'the original Star Wars' → FRANCHISE_LINEAGE.",
            "'documentary' → FORMAT_VISUAL.",
        ),
        (EndpointRoute.KEYWORD,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    CENTRAL_TOPIC = (
        "Central topic / about-ness",
        (
            "A concrete, nameable subject the film centers on — a real or "
            "specific person, event, place, or phenomenon it is ABOUT."
        ),
        (
            "Concrete nameable subjects only. An abstract human experience "
            "or moral essence is the story's shape → STORY_THEMATIC_ARCHETYPE. "
            "Mere presence of a thing → ELEMENT_PRESENCE. The story's "
            "time/place backdrop → NARRATIVE_SETTING."
        ),
        (
            "Test: could you point to the subject as an entry in an "
            "encyclopedia? If yes it is concrete and lands here.",
            "'About X' names the subject; 'has X' is mere presence "
            "(→ ELEMENT_PRESENCE); an abstract condition is thematic "
            "(→ STORY_THEMATIC_ARCHETYPE).",
        ),
        ("about the moon landing", "the sinking of the Titanic", "the Watergate scandal"),
        (
            "'a film exploring loneliness' → STORY_THEMATIC_ARCHETYPE (abstract theme).",
            "'has sharks in it' → ELEMENT_PRESENCE (presence, not subject).",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT,
        CategoryCombineType.ADDITIVE,
    )
    ELEMENT_PRESENCE = (
        "Element / motif presence",
        (
            "A concrete element or motif is merely present in the film — an "
            "object, creature, set piece, or recurring motif — with no claim "
            "about genre, story shape, or character type."
        ),
        (
            "The FALLBACK for a concrete thing that is present but not better "
            "owned elsewhere: not a genre, not the story's shape, not a "
            "character type, not a narrative technique, not sensitive content. "
            "If a sharper category claims it, prefer that. A film centered ON "
            "the thing → CENTRAL_TOPIC. Full event prose → PLOT_EVENTS."
        ),
        (
            "Test: if the only claim is 'this thing appears', it lands here; "
            "if stripping the element leaves a shape, type, or subject, route "
            "there instead.",
            "Bare element nouns read as presence; a multi-clause account of "
            "what happens → PLOT_EVENTS.",
        ),
        ("movies with clowns", "features dragons", "has a shipwreck"),
        (
            "'a rise-from-nothing arc' → STORY_THEMATIC_ARCHETYPE (shape).",
            "'a wise-mentor figure' → CHARACTER_ARCHETYPE (type).",
            "'a gut-punch finale' → EMOTIONAL_EXPERIENTIAL.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT,
        CategoryCombineType.ADDITIVE,
    )
    CHARACTER_ARCHETYPE = (
        "Character archetype",
        (
            "A static character TYPE defined by disposition or role and "
            "carrying no arc — the kind of figure that appears, not what "
            "happens to them."
        ),
        (
            "Fixed type only. A specific named persona → NAMED_CHARACTER. A "
            "trajectory or change-over-time (how a figure rises, falls, or "
            "transforms) is the story's shape → STORY_THEMATIC_ARCHETYPE. "
            "Lead/cast structure (single lead, ensemble) → NARRATIVE_DEVICES."
        ),
        (
            "Test: does the trait name WHAT KIND of person (static) or WHAT "
            "THEY GO THROUGH (an arc)? Static type → here; arc or "
            "transformation → STORY_THEMATIC_ARCHETYPE.",
            "A compound figure stays one trait — do not peel its descriptors "
            "into separate facets.",
        ),
        ("femme fatale", "reluctant hero", "lovable rogue"),
        (
            "'a slow fall from grace' → STORY_THEMATIC_ARCHETYPE (trajectory, not type).",
            "'ensemble cast' → NARRATIVE_DEVICES.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.PREFERRED_REPRESENTATION_FALLBACK,
        CategoryCombineType.ADDITIVE,
    )
    AWARDS = (
        "Award records",
        (
            "Formal award wins or nominations from a named ceremony or body "
            "— the major international film awards (Cannes/Palme d'Or, BAFTA, "
            "Golden Globes, Sundance, and their peers) — including multi-win "
            "superlatives."
        ),
        (
            "A formal ceremony OUTCOME (won/nominated/awarded). A plain "
            "goodness-degree with no award reference → GENERAL_APPEAL. Broad "
            "acclaim or canonical standing → CULTURAL_STATUS. Praise of a "
            "named aspect → SPECIFIC_PRAISE_CRITICISM. Aspirational praise "
            "('award-worthy') → GENERAL_APPEAL; an actual win/nomination "
            "routes here."
        ),
        (
            "Trigger: a named ceremony or awarding body plus an outcome verb "
            "(won/nominated/awarded).",
        ),
        ("Cannes Palme d'Or winner", "BAFTA-nominated", "Golden Globe winner"),
        (
            "'highly acclaimed' → CULTURAL_STATUS.",
            "'award-worthy' → GENERAL_APPEAL (aspirational, not an actual win).",
        ),
        (EndpointRoute.AWARDS,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    TRENDING = (
        "Trending",
        (
            "Live 'right now' status — 'trending', 'what's everyone "
            "watching', 'hot this week', 'currently buzzing', 'popular "
            "right now'."
        ),
        (
            "Live-refreshed signal only. Static popularity → GENERAL_APPEAL. "
            "Recent releases → RELEASE_DATE."
        ),
        (
            "'Right now' / 'this week' / 'currently' is the disambiguator vs static popularity.",
        ),
        ("trending now", "what's everyone watching", "hot right now"),
        (
            "'popular movies' → GENERAL_APPEAL (static).",
            "'recent movies' → RELEASE_DATE.",
        ),
        (EndpointRoute.TRENDING,),
        HandlerBucket.NO_LLM_PURE_CODE,
        CategoryCombineType.SINGLE,
    )

    # -----------------------------------------------------------------
    # Structured single-attribute (one metadata column per category)
    # -----------------------------------------------------------------

    RELEASE_DATE = (
        "Release date / era",
        (
            "Date ranges and eras — '90s', 'old', 'recent', 'before 2000', "
            "'in the 2000s', 'old-school', 'pre-millennium'."
        ),
        (
            "Range/decay framings only. Ordinal position ('newest', 'first', "
            "'latest one') → CHRONOLOGICAL. Live trending → TRENDING."
        ),
        (
            "Vague terms ('modern', 'recent', 'old-school') route here.",
            "'Classic' alone → CULTURAL_STATUS; explicit era words ('old', "
            "'modern') route here as their own traits.",
        ),
        ("from the 90s", "before 2000", "old-school"),
        (
            "'the newest one' → CHRONOLOGICAL.",
            "'trending' → TRENDING.",
            "'classic' → CULTURAL_STATUS (canonical stature, not era word alone).",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    RUNTIME = (
        "Runtime",
        (
            "Movie length — 'around 90 minutes', 'short', 'long', 'under 2 "
            "hours', 'feature-length', 'epic length'."
        ),
        (
            "Fires only when duration is explicit — bare 'movies' carries "
            "no runtime trait. Franchise size ('long-running series') → "
            "FRANCHISE_LINEAGE."
        ),
        (
            "'Short films' splits: 'short' (length) here + 'shorts' "
            "(non-default media format) → MEDIA_TYPE.",
        ),
        ("under 2 hours", "around 90 minutes", "epic length"),
        (
            "Bare 'movies' — no firing.",
            "'long-running franchise' → FRANCHISE_LINEAGE.",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    MATURITY_RATING = (
        "Maturity rating",
        "Rating ceiling/floor — 'PG-13 max', 'rated R', 'G-rated', 'no NC-17'.",
        (
            "Fires ONLY when the trait explicitly names or constrains a "
            "certification rating (G/PG/PG-13/R/NC-17 or an explicit "
            "ceiling/floor). Never a proxy for an audience or a suitability "
            "wish: an audience description → TARGET_AUDIENCE (which already "
            "applies rating suitability internally, so a separate rating call "
            "is redundant). Content-sensitivity → SENSITIVE_CONTENT."
        ),
        (
            "'No R-rated' is a negative-polarity ceiling — still an explicit "
            "rating, routes here.",
            "An audience or suitability ask that names no rating does NOT "
            "route here — never infer a rating ceiling from who a film is for.",
        ),
        ("PG-13 or below", "rated R", "G-rated"),
        (
            "'aimed at grown-ups' → TARGET_AUDIENCE (audience pitch, no explicit rating).",
            "'graphic violence' → SENSITIVE_CONTENT (content axis).",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    AUDIO_LANGUAGE = (
        "Audio language",
        "Original audio language — the spoken track or its subtitling/dubbing ('subtitled', 'dubbed into English', 'in the original Cantonese').",
        (
            "Fires ONLY when the surface form explicitly names a spoken/audio "
            "language or names subtitling or dubbing. A bare nationality or "
            "country adjective NEVER implies audio language — do not infer a "
            "spoken language from where a film is made or its cultural "
            "tradition; those route to COUNTRY_OF_ORIGIN / CULTURAL_TRADITION. "
            "Original audio only — a film's language may not match its "
            "production country or cinematic tradition."
        ),
        (
            "'Subtitled' implies non-English audio.",
        ),
        ("subtitled, not dubbed", "spoken in Mandarin", "in the original Tamil"),
        (
            "'Bollywood' → CULTURAL_TRADITION.",
            "'made in France' → COUNTRY_OF_ORIGIN.",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    STREAMING = (
        "Streaming platform",
        "Movies on a specific streaming platform — 'on Netflix', 'on Hulu', 'on Prime', 'streaming on Max'.",
        (
            "Provider availability only. Studio brand (Disney) → STUDIO_BRAND "
            "even when the studio owns a service."
        ),
        (
            "'Available to stream' without a specific service still routes here as a generic provider gate.",
        ),
        ("on Netflix", "streaming on Hulu", "Prime Video"),
        (
            "'Disney movies' → STUDIO_BRAND.",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    FINANCIAL_SCALE = (
        "Financial scale",
        (
            "Budget AND box-office magnitude — 'big-budget', 'low-budget', "
            "'indie scale', 'shoestring', 'blockbuster', 'flop', 'sleeper "
            "hit', 'made bank'."
        ),
        (
            "Financial axis only. Quality framings ('crowd-pleaser', 'cult "
            "flop') compose with GENERAL_APPEAL / CULTURAL_STATUS separately."
        ),
        (
            "Compound terms span both axes: 'blockbuster' implies big budget "
            "AND big gross; 'indie hit' implies small budget AND outsized "
            "gross. Fires once for the financial axis.",
        ),
        ("blockbuster", "indie scale", "low-budget"),
        (
            "'cult classic' → CULTURAL_STATUS (no financial axis).",
            "'popular' → GENERAL_APPEAL.",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    NUMERIC_RECEPTION_SCORE = (
        "Numeric reception score",
        "Specific numeric reception thresholds — 'rated above 8', '70%+ on RT', '5-star', 'IMDb over 7.5'.",
        (
            "Quality expressed as a NUMBER or rating-system threshold (stars, "
            "points, percent). Quality stated qualitatively without a number "
            "→ GENERAL_APPEAL."
        ),
        (
            "Test: is there a number or a rating-system reference? If not, it "
            "is not this category.",
        ),
        ("rated above 8", "5-star movies", "above 75% on RT"),
        (
            "'highly rated' → GENERAL_APPEAL (no number).",
            "'award-worthy' → GENERAL_APPEAL (aspirational, not a score).",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    COUNTRY_OF_ORIGIN = (
        "Country of origin",
        (
            "The legal/financial production nation — where a film was "
            "produced or financed ('Mexican films', 'produced in', "
            "'British co-production')."
        ),
        (
            "Production nation only. Where it was physically shot → "
            "FILMING_LOCATION. A named cinema tradition/aesthetic → "
            "CULTURAL_TRADITION. The spoken language → AUDIO_LANGUAGE. The "
            "story's in-world place → NARRATIVE_SETTING."
        ),
        (
            "Test: does the trait assert WHERE THE FILM WAS MADE/FINANCED? "
            "Then here. A bare nationality is ambiguous between this and "
            "CULTURAL_TRADITION — favor CULTURAL_TRADITION for "
            "cinema-as-aesthetic phrasing, this for explicit production "
            "framing — and never implies AUDIO_LANGUAGE.",
        ),
        ("Mexican films", "produced in France", "British co-production"),
        (
            "'shot in New Zealand' → FILMING_LOCATION (physical filming).",
            "'Korean cinema' → CULTURAL_TRADITION (named aesthetic).",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    MEDIA_TYPE = (
        "Media type",
        "Non-default media formats — 'TV movies', 'video releases', 'shorts (when explicit)', 'made-for-TV'.",
        (
            "Fires only on explicit non-default request. Bare 'show me "
            "movies' has no media-type trait. Format ('documentary', 'anime') "
            "→ FORMAT_VISUAL."
        ),
        (
            "'Short films' splits: 'shorts' (non-default format) here + "
            "'short' (length) → RUNTIME.",
        ),
        ("TV movies", "shorts", "direct-to-video"),
        (
            "Bare 'movies' — default, no firing.",
            "'documentary' → FORMAT_VISUAL.",
        ),
        (EndpointRoute.MEDIA_TYPE,),
        HandlerBucket.NO_LLM_PURE_CODE,
        CategoryCombineType.SINGLE,
    )

    # -----------------------------------------------------------------
    # Structured / keyword continuing
    # -----------------------------------------------------------------

    GENRE = (
        "Genre",
        (
            "Top-level and sub-genres — horror, action, comedy, sci-fi, "
            "drama, romance, animation, thriller, fantasy, western, neo-noir, "
            "cozy mystery, slasher, space opera, body horror."
        ),
        (
            "Genre identity. Story archetype ('revenge', 'underdog', 'heist', "
            "'post-apocalyptic') → STORY_THEMATIC_ARCHETYPE. Format "
            "('documentary', 'anime') → FORMAT_VISUAL. Pure tonal ('dark', "
            "'whimsical') without genre anchor → EMOTIONAL_EXPERIENTIAL."
        ),
        (
            "'Dark action' splits: action here + dark → EMOTIONAL_EXPERIENTIAL.",
            "Known subgenre compounds stay whole here: 'dark comedy', 'body horror', 'space opera'.",
            "Sub-genre falls back to semantic genre_signatures when no canonical tag exists.",
        ),
        ("horror", "neo-noir", "cozy mystery"),
        (
            "'underdog stories' → STORY_THEMATIC_ARCHETYPE.",
            "'anime' → FORMAT_VISUAL.",
            "'slow burn' → EMOTIONAL_EXPERIENTIAL.",
            "'dark action' splits: 'dark' → EMOTIONAL_EXPERIENTIAL + 'action' here.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.PREFERRED_REPRESENTATION_FALLBACK,
        CategoryCombineType.ALTERNATIVES,
    )
    CULTURAL_TRADITION = (
        "Cultural tradition / national cinema",
        (
            "A named cinema tradition or movement treated as an aesthetic — "
            "Bollywood, Korean cinema, Hong Kong action, Italian neorealism, "
            "French New Wave, Nordic noir, Dogme 95."
        ),
        (
            "A named tradition-as-aesthetic. Where the film was made/financed "
            "→ COUNTRY_OF_ORIGIN. The spoken language → AUDIO_LANGUAGE."
        ),
        (
            "Test: does the trait name a recognized film tradition/movement "
            "(an aesthetic lineage) rather than just a country? A tradition "
            "can diverge from production nation (its films may be "
            "foreign-financed), so prefer this when the aesthetic is named.",
        ),
        ("Bollywood", "Korean cinema", "French New Wave"),
        (
            "'produced in France' → COUNTRY_OF_ORIGIN (production nation).",
            "'in the original Korean' → AUDIO_LANGUAGE (spoken language).",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.METADATA),
        HandlerBucket.PREFERRED_REPRESENTATION_FALLBACK,
        CategoryCombineType.ALTERNATIVES,
    )
    FILMING_LOCATION = (
        "Filming location",
        (
            "Where a film was physically SHOT — 'filmed in New Zealand', "
            "'shot on location in Iceland', 'Morocco shoots'."
        ),
        (
            "Physical filming geography only. Where it was made/financed → "
            "COUNTRY_OF_ORIGIN. The story's in-world place → NARRATIVE_SETTING."
        ),
        (
            "Test: does the trait say where the CAMERAS rolled? Then here — "
            "this can differ from both the production nation and the story's "
            "setting (a film is often shot somewhere it is neither set nor "
            "financed).",
        ),
        ("shot in Iceland", "filmed in Morocco", "on location in Vietnam"),
        (
            "'set in a distant future' → NARRATIVE_SETTING (story world).",
            "'a British co-production' → COUNTRY_OF_ORIGIN (production nation).",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    FORMAT_VISUAL = (
        "Format + visual-format specifics",
        (
            "Format and visual specifics — documentary, anime, mockumentary, "
            "B&W, 70mm, IMAX, found-footage, widescreen, handheld, "
            "single-take long shot, stop-motion."
        ),
        (
            "Form the movie takes, descriptively. Acclaim framing ('visually "
            "stunning', '70mm masterpiece') → VISUAL_CRAFT_ACCLAIM. Genre → "
            "GENRE. Non-default media type ('TV movies') → MEDIA_TYPE."
        ),
        (
            "Surface decides between this and VISUAL_CRAFT_ACCLAIM: 'shot on "
            "70mm' (descriptive) here vs 'IMAX experience' (acclaim) → "
            "VISUAL_CRAFT_ACCLAIM.",
        ),
        ("documentary", "anime", "found-footage"),
        (
            "'visually stunning' → VISUAL_CRAFT_ACCLAIM.",
            "'horror' → GENRE.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.PREFERRED_REPRESENTATION_FALLBACK,
        CategoryCombineType.ALTERNATIVES,
    )
    NARRATIVE_DEVICES = (
        "Narrative devices + structural form + how-told craft",
        (
            "How the story is structured/told at the craft level — plot "
            "twist, nonlinear timeline, unreliable narrator, single-location, "
            "anthology, ensemble, two-hander, POV mechanics, "
            "character-vs-plot focus."
        ),
        (
            "Structural devices only. Pacing-as-experience ('slow burn', "
            "'frenetic') → EMOTIONAL_EXPERIENTIAL. Structural ending types "
            "('twist ending', 'happy ending', 'downer ending') ALSO → "
            "EMOTIONAL_EXPERIENTIAL despite seeming structural. Dialogue "
            "acclaim → DIALOGUE_CRAFT_ACCLAIM."
        ),
        (
            "'Plot twist' (mid-story device) here vs 'twist ending' → "
            "EMOTIONAL_EXPERIENTIAL.",
            "'Sorkin-style' as structural rapid-fire pattern → here; as "
            "praise → DIALOGUE_CRAFT_ACCLAIM. Surface framing decides.",
        ),
        ("nonlinear timeline", "anthology film", "single-location thriller"),
        (
            "'slow burn' → EMOTIONAL_EXPERIENTIAL.",
            "'twist ending' → EMOTIONAL_EXPERIENTIAL.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.PREFERRED_REPRESENTATION_FALLBACK,
        CategoryCombineType.ADDITIVE,
    )
    TARGET_AUDIENCE = (
        "Target audience",
        (
            "The trait names a specific audience — a group of PEOPLE the film "
            "is pitched to or suitable for (by age or demographic)."
        ),
        (
            "Fires when the trait's subject is an explicitly named audience. "
            "This category alone carries the audience ask — its handler "
            "already applies the matching rating/suitability, so do NOT add a "
            "separate MATURITY_RATING or SENSITIVE_CONTENT call for the same "
            "audience. A viewing situation not defined by people → "
            "VIEWING_OCCASION. A story archetype → STORY_THEMATIC_ARCHETYPE."
        ),
        (
            "The audience must be explicitly named — do not infer an audience "
            "from an occasion or a genre.",
            "An explicit rating or an explicit mature-content axis is a "
            "separate trait (MATURITY_RATING / SENSITIVE_CONTENT); a bare "
            "audience pitch is not.",
        ),
        ("for grown-ups", "aimed at older audiences", "grandparent-friendly"),
        (
            "'date night' → VIEWING_OCCASION (occasion, not a named audience).",
            "'a fish-out-of-water story' → STORY_THEMATIC_ARCHETYPE (story shape).",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.METADATA, EndpointRoute.SEMANTIC),
        HandlerBucket.AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST,
        CategoryCombineType.ALTERNATIVES,
    )
    SENSITIVE_CONTENT = (
        "Sensitive content",
        (
            "Presence and intensity of mature or objectionable content the "
            "film CONTAINS — graphic violence, sexual content, nudity, strong "
            "language, drug use, disturbing imagery."
        ),
        (
            "Fires ONLY when the trait is explicitly about the mature content "
            "itself (its presence or intensity). An audience description → "
            "TARGET_AUDIENCE (which handles suitability). An explicit rating "
            "→ MATURITY_RATING. Indexes the content axis itself; an implied "
            "ceiling inferred from who a film is for is not this category."
        ),
        (
            "A content ask framed as avoidance still routes here: it names "
            "the mature-content axis, and present-vs-avoid is tracked "
            "separately as polarity — the call still describes the content, "
            "not its absence.",
        ),
        ("graphic violence", "strong sexual content", "heavy drug use"),
        (
            "'aimed at a younger audience' → TARGET_AUDIENCE (audience pitch, not a content axis).",
            "'rated R' → MATURITY_RATING (explicit rating).",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.METADATA, EndpointRoute.SEMANTIC),
        HandlerBucket.AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST,
        CategoryCombineType.CONSENSUS,
    )
    SEASONAL_HOLIDAY = (
        "Seasonal / holiday",
        (
            "Seasonal/holiday framings — Christmas, Halloween, Thanksgiving, "
            "Fourth of July."
        ),
        (
            "Covers both 'movie for watching AT this season' and 'movie SET "
            "at this season'. Pure narrative setting on a date ('set on "
            "Christmas Eve') without seasonal-viewing framing → "
            "NARRATIVE_SETTING."
        ),
        (
            "'Christmas action movie' splits: Christmas here + action → GENRE.",
        ),
        ("Christmas movies", "Halloween viewing", "Thanksgiving family viewing"),
        (
            "'winter setting' → NARRATIVE_SETTING.",
            "'documentary about Christmas' → CENTRAL_TOPIC (Christmas is the topic).",
            "'snowed-in plot' → PLOT_EVENTS.",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )

    # -----------------------------------------------------------------
    # Semantic-driven
    # -----------------------------------------------------------------

    PLOT_EVENTS = (
        "Plot events",
        (
            "Literal plot beats as transcript-style prose — specific actions "
            "or events that occur inside the film, described concretely."
        ),
        (
            "A specific beat framed as happening WITHIN a larger story. The "
            "same content framed as the film's whole-arc shape (its elevator "
            "pitch) → STORY_THEMATIC_ARCHETYPE. A bare element noun → "
            "ELEMENT_PRESENCE. Time/place backdrop → NARRATIVE_SETTING."
        ),
        (
            "Test: is this the elevator-pitch SHAPE of the whole film, or one "
            "thing that happens inside a larger story? Whole-arc shape → "
            "STORY_THEMATIC_ARCHETYPE; an internal beat → here.",
            "Framing decides, not content — the same event routes either way "
            "depending on whether it is presented as the overall arc or as a "
            "step within it.",
        ),
        (
            "a getaway driver is double-crossed mid-job",
            "a diver is left stranded far offshore",
            "two strangers secretly swap lives for a week",
        ),
        (
            "'a rags-to-riches rise' → STORY_THEMATIC_ARCHETYPE (whole-arc shape).",
            "'has a shipwreck' → ELEMENT_PRESENCE (bare element).",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    NARRATIVE_SETTING = (
        "Narrative setting (time/place)",
        (
            "The story's in-world time and place — 'set in 1940s Berlin', "
            "'during the Cold War', 'a remote island', 'medieval Europe', "
            "'deep space'."
        ),
        (
            "The story's internal time/place ('set in / takes place in / "
            "during'). Production era ('90s movies') → RELEASE_DATE. Where it "
            "was physically shot → FILMING_LOCATION. A concrete focal subject "
            "('about WWII') → CENTRAL_TOPIC."
        ),
        (
            "Test: does the trait describe WHERE/WHEN THE STORY HAPPENS "
            "(in-world)? Then here — distinct from when the film was released "
            "and from where it was shot.",
        ),
        ("set in 1920s Chicago", "takes place in deep space", "during the Cold War"),
        (
            "'90s movies' → RELEASE_DATE (production era).",
            "'about the Cold War' → CENTRAL_TOPIC (concrete subject).",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    STORY_THEMATIC_ARCHETYPE = (
        "Story / thematic archetype",
        (
            "The overall SHAPE or thematic essence of the film — its "
            "elevator-pitch arc, or the abstract human experience at its "
            "core. Captures what kind of story this is at the whole-film level."
        ),
        (
            "The whole-film arc/theme. A specific internal beat (one action "
            "within the larger story) → PLOT_EVENTS. A concrete nameable "
            "subject → CENTRAL_TOPIC. A static character type → "
            "CHARACTER_ARCHETYPE. A genre label, even one that implies a "
            "shape → GENRE."
        ),
        (
            "Test: does this describe the film's overall arc — its elevator "
            "pitch — even when phrased as 'someone does something'? Then here: "
            "an actor performing the arc is still the shape.",
            "One beat inside a larger story → PLOT_EVENTS; a (possibly "
            "qualified) category label → GENRE.",
        ),
        ("rags-to-riches climb", "forbidden romance across a divide", "a slow fall from grace"),
        (
            "'a getaway goes wrong mid-job' → PLOT_EVENTS (an internal beat).",
            "'a quiet character drama' → GENRE (a qualified label).",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.PREFERRED_REPRESENTATION_FALLBACK,
        CategoryCombineType.ADDITIVE,
    )
    EMOTIONAL_EXPERIENTIAL = (
        "Emotional / experiential",
        (
            "All emotional/experiential framings — tone (dark, whimsical, "
            "cozy, eerie, uplifting), cognitive demand (mindless vs "
            "cerebral), pacing-as-experience ('languid', 'frenetic'), "
            "self-experience goals ('make me cry', 'cheer me up'), "
            "comfort-watch ('feel-better movie'), post-viewing resonance "
            "('haunting', 'gut-punch ending', 'forgettable'), structural "
            "ending types ('happy ending', 'downer ending', 'ambiguous "
            "ending')."
        ),
        (
            "Route here ONLY when the affective/experiential quality is the "
            "trait's PRIMARY content. A feeling carried by another axis goes "
            "to that axis: a named genre that connotes a mood is still GENRE; "
            "a viewing situation → VIEWING_OCCASION; a mid-story structural "
            "device → NARRATIVE_DEVICES; a story arc → STORY_THEMATIC_ARCHETYPE."
        ),
        (
            "Test: is the affective quality ITSELF the ask, or merely a "
            "connotation of something with a sharper home? A genre, occasion, "
            "device, or arc that just evokes a feeling routes to its own "
            "category — not here, even when the emotional read looks correct.",
            "Structural ending types live HERE despite being structural — "
            "their emotional weight is what defines them.",
        ),
        ("make me cry", "haunting", "a feel-better comfort watch"),
        (
            "'date night' → VIEWING_OCCASION (a situation, not a feeling).",
            "'a moody noir' → GENRE (a genre that connotes a mood).",
            "'nonlinear timeline' → NARRATIVE_DEVICES (a structural device).",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    VIEWING_OCCASION = (
        "Viewing occasion",
        (
            "A viewing situation or occasion — the setting, moment, or "
            "activity a film is wanted FOR — that is not defined by a named "
            "audience of people."
        ),
        (
            "The trait's subject is the occasion itself. A named audience of "
            "people → TARGET_AUDIENCE. A feeling or emotional state sought "
            "→ EMOTIONAL_EXPERIENTIAL."
        ),
        (
            "A people-word appearing inside an occasion phrase does not "
            "promote it to TARGET_AUDIENCE — route by whether the subject is "
            "the occasion or the audience.",
        ),
        ("date night", "rainy Sunday afternoon", "background while cooking"),
        (
            "'comfort watch' → EMOTIONAL_EXPERIENTIAL (a feeling, not an occasion).",
            "'aimed at grown-ups' → TARGET_AUDIENCE (a named audience).",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    VISUAL_CRAFT_ACCLAIM = (
        "Visual craft acclaim",
        (
            "Acclaim attached to the visual axis — 'visually stunning', "
            "'killer cinematography', 'beautifully shot', 'IMAX-shot', "
            "'practical effects', 'technical marvel', 'gorgeous visuals'."
        ),
        (
            "Acclaim framing only. Named cinematographer/VFX-supervisor → "
            "BELOW_THE_LINE_CREATOR. Descriptive format ('shot in B&W', "
            "'shot on 70mm' without praise) → FORMAT_VISUAL."
        ),
        (
            "'70mm masterpiece' splits: 'masterpiece' here + '70mm' → "
            "FORMAT_VISUAL.",
        ),
        ("visually stunning", "killer cinematography", "practical effects"),
        (
            "'Roger Deakins' → BELOW_THE_LINE_CREATOR.",
            "'shot in B&W' → FORMAT_VISUAL.",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    MUSIC_SCORE_ACCLAIM = (
        "Music / score acclaim",
        (
            "Acclaim for the music side — 'iconic score', 'great "
            "soundtrack', 'memorable theme', 'amazing music', 'killer "
            "needle drops'."
        ),
        (
            "Acclaim only. Named composer (John Williams, Hans Zimmer) is "
            "indexed → PERSON_CREDIT. Genre 'musical' → GENRE."
        ),
        (
            "'The iconic John Williams score' splits: 'John Williams' → "
            "PERSON_CREDIT + 'iconic score' here.",
        ),
        ("iconic score", "memorable theme music", "killer needle drops"),
        (
            "'John Williams' → PERSON_CREDIT.",
            "'musicals' → GENRE.",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    DIALOGUE_CRAFT_ACCLAIM = (
        "Dialogue craft acclaim",
        (
            "Acclaim for dialogue — 'quotable dialogue', 'Sorkin-style', "
            "'naturalistic dialogue', 'snappy banter', 'razor-sharp writing'."
        ),
        (
            "Dialogue acclaim only. Structural rapid-fire pattern (as a "
            "how-told device) → NARRATIVE_DEVICES. Writer credit → "
            "PERSON_CREDIT."
        ),
        (
            "'Sorkin-style' as praise for the writing → here; as a "
            "structural rapid-fire pattern → NARRATIVE_DEVICES. Surface "
            "framing decides.",
        ),
        ("snappy dialogue", "quotable lines", "Sorkin-style"),
        (
            "'Aaron Sorkin movies' → PERSON_CREDIT.",
            "'rapid-fire structure' → NARRATIVE_DEVICES.",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    GENERAL_APPEAL = (
        "General appeal / quality baseline",
        (
            "Quality as a DEGREE of goodness or popularity, stated "
            "qualitatively with no number — 'well-received', 'highly "
            "regarded', 'popular', 'great', 'crowd-pleaser'."
        ),
        (
            "Goodness-degree without a number. A numeric/rating threshold → "
            "NUMERIC_RECEPTION_SCORE. The work's canonical position or "
            "reception SHAPE (its standing in the culture) → CULTURAL_STATUS. "
            "Praise/criticism tied to a named aspect → "
            "SPECIFIC_PRAISE_CRITICISM. Formal award outcomes → AWARDS. Live "
            "right-now popularity → TRENDING."
        ),
        (
            "Test: is this just 'how good or popular' — no number, no named "
            "aspect, no claim about cultural standing? Then here.",
        ),
        ("well-received", "highly regarded", "crowd-pleaser"),
        (
            "'rated above 8' → NUMERIC_RECEPTION_SCORE (a number).",
            "'a cult classic' → CULTURAL_STATUS (cultural standing).",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
    CULTURAL_STATUS = (
        "Cultural status / canonical stature",
        (
            "The work's standing in the culture and its reception SHAPE — "
            "'classic', 'cult classic', 'overhyped', 'divisive', "
            "'era-defining', 'still holds up', 'influential', 'iconic', "
            "'landmark', 'ahead of its time'."
        ),
        (
            "The whole-work's cultural position, not a part liked or disliked "
            "and not a plain goodness-degree. Praise/criticism of a named "
            "aspect → SPECIFIC_PRAISE_CRITICISM. Goodness-degree "
            "('well-received', 'popular') → GENERAL_APPEAL. A numeric "
            "threshold → NUMERIC_RECEPTION_SCORE. Formal award outcomes → "
            "AWARDS."
        ),
        (
            "Test: does the trait place the whole film in the culture (its "
            "canon, legacy, or how reception split) rather than rate how good "
            "it is? Then here.",
            "'Classic' alone lives here; add RELEASE_DATE only when an "
            "explicit era word is also present.",
        ),
        ("classic", "cult classic", "era-defining", "still holds up", "divisive"),
        (
            "'praised for its taut pacing' → SPECIFIC_PRAISE_CRITICISM (named aspect).",
            "'well-received' → GENERAL_APPEAL (goodness-degree).",
        ),
        (EndpointRoute.SEMANTIC, EndpointRoute.METADATA),
        HandlerBucket.SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT,
        CategoryCombineType.ADDITIVE,
    )
    SPECIFIC_PRAISE_CRITICISM = (
        "Specific praise / criticism",
        (
            "Reception prose tied to a SPECIFIC part, quality, or aspect "
            "people liked or disliked — 'praised for its tension', "
            "'criticized as plodding', 'loved for its dialogue', 'criticized "
            "for a weak ending'."
        ),
        (
            "Aspect-level praise/criticism only. The whole-work's cultural "
            "standing or reception shape → CULTURAL_STATUS. A plain "
            "goodness-degree → GENERAL_APPEAL. A numeric threshold → "
            "NUMERIC_RECEPTION_SCORE. Formal award outcomes → AWARDS."
        ),
        (
            "Test: does the praise/criticism attach to a NAMED aspect "
            "(pacing, tension, dialogue, ending, score, script)? Then here. "
            "If it rates the whole film's standing instead → CULTURAL_STATUS.",
        ),
        (
            "praised for its tension",
            "criticized as plodding",
            "loved for its dialogue",
            "criticized for a weak ending",
        ),
        (
            "'a cult classic' → CULTURAL_STATUS (whole-work standing).",
            "'highly rated' → GENERAL_APPEAL (goodness-degree, no named aspect).",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )

    # -----------------------------------------------------------------
    # Trick / specialized
    # -----------------------------------------------------------------

    BELOW_THE_LINE_CREATOR = (
        "Below-the-line creator",
        (
            "Literal below-the-line credit lookups — cinematographer, "
            "editor, production designer, costume designer, VFX supervisor. "
            "'Roger Deakins', 'Thelma Schoonmaker-edited', 'Sandy Powell "
            "costumes', 'Colleen Atwood designs' framed as a direct credit "
            "ask."
        ),
        (
            "RESERVED — currently returns empty until backing data lands. "
            "Routing here keeps literal credit lookups from misrouting to "
            "PERSON_CREDIT (which would silently return zero results for "
            "non-indexed roles). Stylistic-transfer framings ('Roger "
            "Deakins-style cinematography', 'Deakins-shot' as praise) are "
            "resolved parametrically into concrete craft attributes upstream "
            "and route to VISUAL_CRAFT_ACCLAIM or FORMAT_VISUAL — they do "
            "not reach this category. Director, writer, actor, producer, "
            "composer remain at PERSON_CREDIT."
        ),
        (
            "Surface form must be a direct credit ask, not a stylistic "
            "comparison. 'Roger Deakins movies' fires here; 'Roger "
            "Deakins-style' does not.",
        ),
        ("Roger Deakins movies", "Sandy Powell costumes", "edited by Thelma Schoonmaker"),
        (
            "'Christopher Nolan' → PERSON_CREDIT (director, indexed).",
            "'great cinematography' (no name) → VISUAL_CRAFT_ACCLAIM.",
            "'Roger Deakins-style cinematography' → resolved upstream into "
            "concrete craft attributes; routes to VISUAL_CRAFT_ACCLAIM.",
        ),
        (),
        HandlerBucket.EXPLICIT_NO_OP,
        CategoryCombineType.NO_OP,
    )
    NAMED_SOURCE_CREATOR = (
        "Named source creator",
        (
            "Named creator of source material being adapted — Stephen "
            "King, Tolkien, Shakespeare, Philip K. Dick, Neil Gaiman, Jane "
            "Austen."
        ),
        (
            "Source-material authors only. A character-anchored franchise "
            "named as the source ('Sherlock Holmes books') splits — "
            "character → CHARACTER_FRANCHISE, medium → ADAPTATION_SOURCE; "
            "Sherlock himself is the book's subject, not its author. A "
            "character-less franchise as source ('Star Wars novelizations') "
            "→ FRANCHISE_LINEAGE for the franchise + ADAPTATION_SOURCE for "
            "the medium. Generic medium with no named creator → "
            "ADAPTATION_SOURCE alone. Film credit (director, writer, "
            "actor) → PERSON_CREDIT."
        ),
        (
            "'X's <medium>' phrases split into two traits: creator here + "
            "medium ('books', 'plays', 'novels') → ADAPTATION_SOURCE.",
            "A named referent stays as one trait — never split mid-name "
            "('Stephen King' is one trait).",
        ),
        ("Stephen King novels", "Shakespeare adaptations", "Neil Gaiman"),
        (
            "'Sherlock Holmes books' splits: 'Sherlock Holmes' (book's "
            "subject) → CHARACTER_FRANCHISE + 'books' → ADAPTATION_SOURCE.",
            "'Christopher Nolan' → PERSON_CREDIT (director).",
        ),
        # Routes to ENTITY as a writer-credit lookup. IMDB's GraphQL
        # `writer` category aggregates every writing credit (screenplay,
        # story, novel, characters, adaptation) into one bucket, and an
        # audit found near-100% recall on canonical adaptations for
        # distinctively named source authors. The schema dispatch
        # (endpoint_registry._ENTITY_DISPATCH) hands this category
        # WriterOnlyPersonQuerySpec, which coerces person_category to
        # WRITER so the lookup always hits the writer posting table.
        (EndpointRoute.ENTITY,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )

    # -----------------------------------------------------------------
    # Ordinal selection
    # -----------------------------------------------------------------

    CHRONOLOGICAL = (
        "Chronological ordinal",
        (
            "Release-date position phrasing — 'first', 'last', 'earliest', "
            "'latest', 'most recent', 'the newest one', 'the oldest one'. "
            "Scores the candidate pool on a continuous recency percentile "
            "curve where every distinct date matters."
        ),
        (
            "Position phrasings only. Range/window framings ('90s', "
            "'recent', 'before 2000') → RELEASE_DATE — those saturate above "
            "a cap; this curve does not. Quality superlatives split: "
            "'best' → GENERAL_APPEAL; 'most acclaimed' → CULTURAL_STATUS."
        ),
        (
            "'The latest Scorsese' splits: 'latest' here + 'Scorsese' → "
            "PERSON_CREDIT.",
        ),
        ("the newest one", "earliest film", "most recent release"),
        (
            "'recent movies' → RELEASE_DATE.",
            "'best of all time' splits: 'best' → GENERAL_APPEAL + 'of all "
            "time' → CULTURAL_STATUS.",
        ),
        (EndpointRoute.CHRONOLOGICAL,),
        HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
        CategoryCombineType.SINGLE,
    )
