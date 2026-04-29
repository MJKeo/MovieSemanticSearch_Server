"""
Trait category taxonomy for the search v2 step-2 grounding LLM.

`CategoryName` is the canonical vocabulary the step-2 pre-pass picks
from when grounding each captured meaning of a query fragment. Each
member carries:

- A descriptive name (the string value emitted to downstream code).
- A compact `description` of what the category is about.
- A `boundary` statement listing what the category does NOT own and
  the explicit redirects to the categories that DO.
- A tuple of `edge_cases` — concrete situations where the LLM is
  most likely to misroute, with the disambiguating cue.
- A tuple of `good_examples` — short trait surface forms that
  clearly belong here. Each example is chosen to illustrate a
  *different* pattern (do not duplicate principles).
- A tuple of `bad_examples` — short trait surface forms that look
  like they belong here but route elsewhere, with the redirect
  spelled out.
- An ordered tuple of `EndpointRoute` values naming the route family
  the category may dispatch to. Endpoint order is priority order:
  for tiered categories the first entry is the authoritative tier and
  later entries are fallbacks; for single-endpoint or combo categories
  the order reflects the primary channel first.
- A `HandlerBucket` identifying the orchestration shape (single /
  mutex / tiered / combo).

All five text-shaped fields (description, boundary, edge_cases,
good_examples, bad_examples) are programmatically inserted into the
step-2 system prompt — wording is kept tight to avoid prompt bloat.

The 45-category list and its underlying granularity principles are
documented in `search_improvement_planning/query_categories.md` and
`search_improvement_planning/v3_category_attributes.md`. Those docs
are the source of truth for routing semantics; this enum mirrors
those definitions in a form the LLM can pick from.
"""

from enum import Enum

from schemas.enums import EndpointRoute, HandlerBucket


class CategoryName(str, Enum):
    description: str
    boundary: str
    edge_cases: tuple[str, ...]
    good_examples: tuple[str, ...]
    bad_examples: tuple[str, ...]
    endpoints: tuple["EndpointRoute", ...]
    bucket: "HandlerBucket"

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
        return obj

    # -----------------------------------------------------------------
    # Structured / lexical
    # -----------------------------------------------------------------

    PERSON_CREDIT = (
        "Person credit",
        "Indexed film credits — actor, director, writer, producer, composer.",
        (
            "Indexed roles only. Cinematographer, editor, production/costume "
            "designer, VFX supervisor → BELOW_THE_LINE_CREATOR. Title strings "
            "→ TITLE_TEXT. Source-material authors (novelists, playwrights) "
            "→ NAMED_SOURCE_CREATOR."
        ),
        (
            "Composer is indexed: 'John Williams score' routes here, not MUSIC_SCORE_ACCLAIM.",
            "Role markers ('starring', 'directed by') are absorbed into the trait, not split out.",
        ),
        ("starring Tom Hanks", "directed by Greta Gerwig", "Hans Zimmer score"),
        (
            "'Roger Deakins-shot' → BELOW_THE_LINE_CREATOR (DP not indexed).",
            "'Stephen King movies' → NAMED_SOURCE_CREATOR (novelist, not film credit).",
        ),
        (EndpointRoute.ENTITY,),
        HandlerBucket.SINGLE,
    )
    TITLE_TEXT = (
        "Title text lookup",
        "Substring matches against the movie's title.",
        (
            "Title-string framing only. Person names → PERSON_CREDIT. "
            "Franchise/character names go to whichever name category fits "
            "the entity (FRANCHISE_LINEAGE for character-less franchises, "
            "CHARACTER_FRANCHISE for character-anchored franchises, "
            "NAMED_CHARACTER for non-anchoring characters) even when the "
            "name overlaps with title fragments."
        ),
        (
            "Disambiguator is the framing: 'in the title' / 'called' / 'titled' signals "
            "this category. Bare names default to entity routing.",
        ),
        ("any movie called Inception", "movies with 'Star' in the title", "titles starting with The"),
        (
            "'Star Wars' → FRANCHISE_LINEAGE (franchise framing dominates).",
            "'Batman' → CHARACTER_FRANCHISE (character-anchored franchise).",
            "'Tom Hanks' → PERSON_CREDIT (person name).",
        ),
        (EndpointRoute.ENTITY,),
        HandlerBucket.SINGLE,
    )
    NAMED_CHARACTER = (
        "Named character",
        (
            "Specific named persona appearing in films but NOT anchoring "
            "their own film franchise — Yoda, Hermione Granger, Severus "
            "Snape, Loki, Aragorn."
        ),
        (
            "Non-anchoring characters only. Character-anchored franchises "
            "(Batman, Bond, Spider-Man, Sherlock Holmes, Harry Potter) → "
            "CHARACTER_FRANCHISE. Static character TYPES (anti-hero, femme "
            "fatale) → CHARACTER_ARCHETYPE. Actors playing the character → "
            "PERSON_CREDIT."
        ),
        (
            "Detection rule for anchoring vs non-anchoring: does a Wikipedia "
            "'List of [X] films' entity exist with this character as the "
            "through-line? Bond/Batman/Spider-Man — yes → CHARACTER_FRANCHISE. "
            "Yoda/Hermione/Loki — no → here.",
            "'Yoda in Star Wars' is two traits: 'Yoda' → here + "
            "'Star Wars' → FRANCHISE_LINEAGE.",
        ),
        ("movies featuring Yoda", "Hermione Granger scenes", "Loki appearances"),
        (
            "'Batman' → CHARACTER_FRANCHISE (anchors a film franchise).",
            "'anti-hero protagonist' → CHARACTER_ARCHETYPE (type, not specific name).",
            "'Daniel Radcliffe' → PERSON_CREDIT (actor not character).",
        ),
        (EndpointRoute.ENTITY,),
        HandlerBucket.SINGLE,
    )
    STUDIO_BRAND = (
        "Studio / brand",
        (
            "Production studios and curated brands — Disney, Pixar, A24, "
            "Studio Ghibli, Blumhouse, Marvel Studios, Dreamworks, Hammer Films."
        ),
        (
            "The entity that made the movie. Franchises the studio owns → "
            "FRANCHISE_LINEAGE. Cultural traditions (Bollywood, Korean cinema) "
            "→ CULTURAL_TRADITION."
        ),
        (
            "Some names are both a studio and a franchise (Marvel = Marvel "
            "Studios vs the MCU; Disney = the studio vs the Disney character "
            "canon). Pick the reading that best fits the surrounding query "
            "context — there is no universal default; lean on whichever entity "
            "is the more natural referent for the phrasing.",
            "Pixar is its own brand even though Disney owns it — emit Pixar here.",
        ),
        ("A24 horror", "Studio Ghibli movies", "Blumhouse"),
        (
            "'Marvel Cinematic Universe' → FRANCHISE_LINEAGE (franchise reading).",
            "'Bollywood' → CULTURAL_TRADITION.",
        ),
        (EndpointRoute.STUDIO,),
        HandlerBucket.SINGLE,
    )
    FRANCHISE_LINEAGE = (
        "Franchise / universe lineage",
        (
            "Two things: membership in a character-less named "
            "franchise/universe (Marvel Cinematic Universe, Star Wars, Lord "
            "of the Rings, Fast & Furious, Star Trek, Mission Impossible) "
            "AND lineage positioning (sequel, prequel, spinoff, reboot, "
            "mainline vs offshoot, original vs remake)."
        ),
        (
            "Lineage POSITIONING fires here regardless of franchise type — "
            "'Batman spinoffs' splits 'spinoffs' to here even though "
            "'Batman' itself goes to CHARACTER_FRANCHISE. Character-anchored "
            "franchise NAMES (Batman, Bond, Spider-Man, Sherlock Holmes, "
            "Harry Potter) → CHARACTER_FRANCHISE. When a generic medium is "
            "named alongside a character-less franchise, the phrase splits "
            "into two traits — e.g. 'Star Wars novelizations' is "
            "'Star Wars' → here + 'novelizations' → ADAPTATION_SOURCE. "
            "Source-material creators (Stephen King, Shakespeare) → "
            "NAMED_SOURCE_CREATOR."
        ),
        (
            "'The original, not the remake' is lineage positioning here, not ADAPTATION_SOURCE.",
            "Marvel Cinematic Universe subgroups ('phase 4') and Star Wars subgroups "
            "('the prequels') route here.",
        ),
        ("Marvel Cinematic Universe movies", "the Star Wars prequels", "Fast & Furious sequels"),
        (
            "'Batman' → CHARACTER_FRANCHISE (character-anchored franchise name).",
            "'based on a comic book' → ADAPTATION_SOURCE (no named franchise).",
        ),
        (EndpointRoute.FRANCHISE_STRUCTURE,),
        HandlerBucket.SINGLE,
    )
    CHARACTER_FRANCHISE = (
        "Character-franchise",
        (
            "Named characters whose identity anchors a film franchise — "
            "Batman, James Bond, Spider-Man, Wolverine, Indiana Jones, "
            "John Wick, Jason Bourne, Sherlock Holmes, Harry Potter, Barbie."
        ),
        (
            "Names the anchoring character only. Sequel/prequel/spinoff/"
            "reboot positioning is a SEPARATE trait → FRANCHISE_LINEAGE. "
            "Non-anchoring characters (Yoda, Hermione, Loki) → "
            "NAMED_CHARACTER. Character-less franchises (Star Wars, Lord "
            "of the Rings, Fast & Furious) → FRANCHISE_LINEAGE."
        ),
        (
            "Detection rule: a Wikipedia 'List of [X] films' entity exists "
            "with the character as through-line. Bond/Batman/Spider-Man — "
            "yes. Yoda/Hermione/Loki — no.",
            "'Sherlock Holmes books' is two traits: 'Sherlock Holmes' → "
            "here + 'books' → ADAPTATION_SOURCE.",
            "'Batman spinoffs' is two traits: 'Batman' → here + 'spinoffs' "
            "→ FRANCHISE_LINEAGE (lineage positioning).",
        ),
        ("Batman", "James Bond", "Barbie"),
        (
            "'Yoda appearances' → NAMED_CHARACTER (non-anchoring).",
            "'Star Wars movies' → FRANCHISE_LINEAGE (character-less).",
            "'Daniel Craig as Bond' is two traits: 'Daniel Craig' → "
            "PERSON_CREDIT + 'Bond' → here.",
        ),
        (EndpointRoute.ENTITY, EndpointRoute.FRANCHISE_STRUCTURE),
        HandlerBucket.COMBO,
    )
    ADAPTATION_SOURCE = (
        "Adaptation source flag",
        (
            "Origin medium of the source material — novel, comic book, true "
            "story, biography, video-game adaptation, remake (as origin medium, "
            "not lineage positioning)."
        ),
        (
            "About WHAT the source medium is. When the query also names a "
            "specific source, the named source becomes a SEPARATE trait — "
            "the character or franchise routes to CHARACTER_FRANCHISE / "
            "FRANCHISE_LINEAGE, the creator routes to NAMED_SOURCE_CREATOR. "
            "Lineage positioning (original vs remake) stays in "
            "FRANCHISE_LINEAGE."
        ),
        (
            "'Stephen King novels' is two traits: 'Stephen King' → "
            "NAMED_SOURCE_CREATOR + 'novels' → here.",
            "'Sherlock Holmes books' is two traits: 'Sherlock Holmes' "
            "(book's subject) → CHARACTER_FRANCHISE + 'books' → here.",
            "'JFK biopic' is two traits: 'JFK' → CENTRAL_TOPIC + "
            "'biopic' → here.",
        ),
        ("novel adaptation", "based on a comic book", "based on a true story"),
        (
            "'the original Star Wars' → FRANCHISE_LINEAGE (lineage position).",
            "'documentary' → FORMAT_VISUAL (format, not source flag).",
        ),
        (EndpointRoute.KEYWORD,),
        HandlerBucket.SINGLE,
    )
    CENTRAL_TOPIC = (
        "Central topic / about-ness",
        (
            "Concrete subject the film is ABOUT — JFK, Vietnam War, Titanic, "
            "Watergate, Princess Diana, the moon landing."
        ),
        (
            "Concrete subjects only. Thematic essence (grief, redemption, "
            "found family) → STORY_THEMATIC_ARCHETYPE. 'Has X' framing → "
            "ELEMENT_PRESENCE. Narrative time/place setting → NARRATIVE_SETTING."
        ),
        (
            "'About' vs 'has' is the disambiguator: 'movies about sharks' (subject) vs "
            "'shark movies' (element-presence).",
            "'Set during Vietnam' is setting → NARRATIVE_SETTING; 'about Vietnam' is centrality.",
        ),
        ("about the moon landing", "Watergate movie", "Princess Diana biopic"),
        (
            "'movies about grief' → STORY_THEMATIC_ARCHETYPE (thematic, not concrete).",
            "'movies with sharks' → ELEMENT_PRESENCE (presence, not centrality).",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.TIERED,
    )
    ELEMENT_PRESENCE = (
        "Element / motif presence",
        (
            "Concrete element appears in the story — 'has clowns', 'zombie "
            "movies', 'shark movies', 'robots', 'movies with horses', 'has a heist'."
        ),
        (
            "'Has X' / mere presence framing. Centrality framing → CENTRAL_TOPIC. "
            "Character types → CHARACTER_ARCHETYPE. Thematic abstraction → "
            "STORY_THEMATIC_ARCHETYPE. Structural devices → NARRATIVE_DEVICES."
        ),
        (
            "Bare-noun 'heist movies' / 'zombie movies' read as element-presence; full plot "
            "description ('a heist crew unravels…') → PLOT_EVENTS.",
        ),
        ("movies with clowns", "zombie movies", "has horses"),
        (
            "'underdog stories' → STORY_THEMATIC_ARCHETYPE.",
            "'anti-hero' → CHARACTER_ARCHETYPE.",
            "'twist ending' → EMOTIONAL_EXPERIENTIAL.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.TIERED,
    )
    CHARACTER_ARCHETYPE = (
        "Character archetype",
        (
            "Static character TYPES — anti-hero, femme fatale, lovable rogue, "
            "reluctant hero, manic pixie dream girl, love-to-hate villain, "
            "underdog protagonist."
        ),
        (
            "Static type only. Specific named personas → NAMED_CHARACTER. "
            "Story shape / character trajectory (redemption arc, coming-of-age) "
            "→ STORY_THEMATIC_ARCHETYPE. Structural single-lead/ensemble framing "
            "→ NARRATIVE_DEVICES."
        ),
        (
            "'Lone female protagonist' is two traits: 'female protagonist' "
            "→ here + 'lone' → NARRATIVE_DEVICES (single-lead structural form).",
        ),
        ("anti-hero protagonist", "femme fatale", "reluctant hero"),
        (
            "'redemption arc' → STORY_THEMATIC_ARCHETYPE (trajectory).",
            "'ensemble cast' → NARRATIVE_DEVICES (structural).",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.TIERED,
    )
    AWARDS = (
        "Award records",
        (
            "Formal award wins/nominations — Oscar, BAFTA, Palme d'Or, "
            "Cannes, Sundance, Golden Globes, multi-win superlatives."
        ),
        (
            "Structured ceremony/outcome data only. Qualitative quality without "
            "award reference → GENERAL_APPEAL. Broad reputation / canon "
            "language ('acclaimed', 'classic') → CULTURAL_STATUS. Specific "
            "aspect praise ('praised for pacing') → SPECIFIC_PRAISE_CRITICISM. "
            "Aspirational praise ('Oscar-worthy') → GENERAL_APPEAL — actual "
            "outcome ('won an Oscar', 'Oscar-nominated') routes here."
        ),
        (
            "Any specific ceremony name (Oscars, Cannes, BAFTAs, Globes) paired with "
            "an outcome verb (won/nominated/awarded) is the routing trigger.",
        ),
        ("Academy Award winners", "Cannes Palme d'Or", "Oscar-nominated for Best Director"),
        (
            "'highly acclaimed' → CULTURAL_STATUS (broad reputation, no formal award).",
            "'Oscar-worthy' → GENERAL_APPEAL (aspirational, not an actual win).",
        ),
        (EndpointRoute.AWARDS,),
        HandlerBucket.SINGLE,
    )
    TRENDING = (
        "Trending",
        (
            "Live 'right now' status — 'trending', 'what's everyone watching', "
            "'hot this week', 'currently buzzing', 'popular right now'."
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
            "'recent movies' → RELEASE_DATE (release-date range).",
        ),
        (EndpointRoute.TRENDING,),
        HandlerBucket.SINGLE,
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
            "'Classic' by itself routes to CULTURAL_STATUS "
            "(canonical stature). Explicit era words such as 'old' or "
            "'modern' route here as their own traits.",
        ),
        ("from the 90s", "before 2000", "old-school"),
        (
            "'the newest one' → CHRONOLOGICAL (ordinal).",
            "'trending' → TRENDING (live).",
            "'classic' → CULTURAL_STATUS (canonical stature, not an era word by itself).",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    RUNTIME = (
        "Runtime",
        (
            "Movie length framings — 'around 90 minutes', 'short', 'long', "
            "'under 2 hours', 'feature-length', 'epic length'."
        ),
        (
            "Fires only when the query mentions duration explicitly — bare "
            "'movies' carries no runtime trait. Franchise size ('long-running "
            "series') → FRANCHISE_LINEAGE."
        ),
        (
            "'Short films' surfaces two traits: 'short' (length) → here + "
            "'shorts' (non-default media format) → MEDIA_TYPE.",
        ),
        ("under 2 hours", "around 90 minutes", "epic length"),
        (
            "Bare 'movies' — no runtime mention, no firing.",
            "'long-running franchise' → FRANCHISE_LINEAGE (franchise size).",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    MATURITY_RATING = (
        "Maturity rating",
        "Rating ceiling/floor — 'PG-13 max', 'rated R', 'G-rated', 'no NC-17'.",
        (
            "Fires only when the query explicitly names a rating "
            "(G/PG/PG-13/R/NC-17). Audience-pitch framing ('family movies') "
            "→ TARGET_AUDIENCE. Content-sensitivity ('no gore') → "
            "SENSITIVE_CONTENT. Phrasing like 'PG-13 family movies' is two "
            "traits: the rating → here, the audience pitch → TARGET_AUDIENCE."
        ),
        (
            "'No R-rated' is a negative-polarity ceiling, still routes here.",
        ),
        ("PG-13 or below", "rated R", "G-rated"),
        (
            "'family-friendly' → TARGET_AUDIENCE (no explicit rating named).",
            "'no gore' → SENSITIVE_CONTENT.",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    AUDIO_LANGUAGE = (
        "Audio language",
        "Original audio language — 'in Korean', 'Spanish-language', 'subtitled', 'French-original'.",
        (
            "Original audio language only. Production country → "
            "COUNTRY_OF_ORIGIN. Cultural tradition → CULTURAL_TRADITION. A "
            "Korean-language film may not be Korean-cinema and vice versa."
        ),
        (
            "'Subtitled' implies non-English audio.",
        ),
        ("Korean-language", "Spanish-language films", "in Japanese with subtitles"),
        (
            "'Bollywood' → CULTURAL_TRADITION.",
            "'made in France' → COUNTRY_OF_ORIGIN.",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    STREAMING = (
        "Streaming platform",
        "Movies available on a specific streaming platform — 'on Netflix', 'on Hulu', 'on Prime', 'streaming on Max'.",
        (
            "Provider availability only. Studio brand (e.g. Disney) → "
            "STUDIO_BRAND even when the studio owns a service."
        ),
        (
            "'Available to stream' without a specific service still routes here as a generic provider gate.",
        ),
        ("on Netflix", "streaming on Hulu", "Prime Video"),
        (
            "'Disney movies' → STUDIO_BRAND.",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    FINANCIAL_SCALE = (
        "Financial scale",
        (
            "Budget AND box-office magnitude — 'big-budget', 'low-budget', "
            "'indie scale', 'shoestring', 'blockbuster', 'flop', 'sleeper hit', "
            "'made bank'."
        ),
        (
            "Financial axis only. Quality framings ('crowd-pleaser', 'cult flop') "
            "compose with GENERAL_APPEAL / CULTURAL_STATUS separately — "
            "those don't fold in here."
        ),
        (
            "Compound terms span both budget and gross: 'blockbuster' "
            "implies big budget AND big gross; 'indie hit' implies small "
            "budget AND outsized gross. The trait fires once for the "
            "financial axis.",
        ),
        ("blockbuster", "indie scale", "low-budget"),
        (
            "'cult classic' → CULTURAL_STATUS (canonical reception, "
            "no financial axis).",
            "'popular' → GENERAL_APPEAL.",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    NUMERIC_RECEPTION_SCORE = (
        "Numeric reception score",
        "Specific numeric reception thresholds — 'rated above 8', '70%+ on RT', '5-star', 'IMDb over 7.5'.",
        (
            "Numeric framing only. Qualitative quality ('well-rated', 'best', "
            "'highly regarded') → GENERAL_APPEAL."
        ),
        (
            "Surface form must contain a number or rating-system reference (stars, percent).",
        ),
        ("rated above 8", "5-star movies", "above 75% on RT"),
        (
            "'highly rated' → GENERAL_APPEAL.",
            "'Oscar-worthy' → GENERAL_APPEAL.",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    COUNTRY_OF_ORIGIN = (
        "Country of origin",
        (
            "Legal/financial production country — 'produced in', 'American films', "
            "'British production', 'Canadian co-production'."
        ),
        (
            "Production country only. Filming geography → FILMING_LOCATION. "
            "Cultural tradition → CULTURAL_TRADITION. When a tradition tag exists, "
            "country is misleading (Hollywood-funded HK action carries US "
            "country_of_origin)."
        ),
        (
            "Bare 'French / Korean / Japanese movies' is ambiguous between this, AUDIO_LANGUAGE, "
            "and CULTURAL_TRADITION — favor CULTURAL_TRADITION for cinema-as-aesthetic phrasing, "
            "this for explicit production framing.",
        ),
        ("produced in France", "American films", "British production"),
        (
            "'filmed in New Zealand' → FILMING_LOCATION.",
            "'Korean cinema' → CULTURAL_TRADITION.",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    MEDIA_TYPE = (
        "Media type",
        "Non-default media formats — 'TV movies', 'video releases', 'shorts (when explicit)', 'made-for-TV'.",
        (
            "Fires only on explicit non-default request. Bare 'show me "
            "movies' has no media-type trait. Format like 'documentary' / "
            "'anime' → FORMAT_VISUAL."
        ),
        (
            "'Short films' surfaces two traits: 'shorts' (non-default media "
            "format) → here + 'short' (length) → RUNTIME.",
        ),
        ("TV movies", "shorts", "direct-to-video"),
        (
            "Bare 'movies' — default, no firing.",
            "'documentary' → FORMAT_VISUAL.",
        ),
        (EndpointRoute.MEDIA_TYPE,),
        HandlerBucket.SINGLE,
    )

    # -----------------------------------------------------------------
    # Structured / keyword continuing
    # -----------------------------------------------------------------

    GENRE = (
        "Genre",
        (
            "Top-level and sub-genres — horror, action, comedy, sci-fi, drama, "
            "romance, animation, thriller, fantasy, western, neo-noir, cozy "
            "mystery, slasher, space opera, body horror."
        ),
        (
            "Genre identity. Story archetype ('revenge', 'underdog', 'heist', "
            "'post-apocalyptic') → STORY_THEMATIC_ARCHETYPE. Format ('documentary', "
            "'anime') → FORMAT_VISUAL. Pure tonal ('dark', 'whimsical') without "
            "genre anchor → EMOTIONAL_EXPERIENTIAL."
        ),
        (
            "Compound 'dark action' splits: GENRE (action) + EMOTIONAL_EXPERIENTIAL (dark).",
            "Known subgenre compounds stay whole here: 'dark comedy', 'body horror', 'space opera'.",
            "Sub-genre falls back to semantic genre_signatures when no canonical tag exists.",
        ),
        ("horror", "neo-noir", "cozy mystery"),
        (
            "'underdog stories' → STORY_THEMATIC_ARCHETYPE.",
            "'anime' → FORMAT_VISUAL.",
            "'slow burn' → EMOTIONAL_EXPERIENTIAL.",
            "'dark action' is two traits: 'dark' → EMOTIONAL_EXPERIENTIAL + 'action' → GENRE.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.MUTEX,
    )
    CULTURAL_TRADITION = (
        "Cultural tradition / national cinema",
        (
            "Named cinema traditions/movements — Bollywood, Korean cinema, "
            "Hong Kong action, Italian neorealism, French New Wave, Nordic noir, "
            "J-horror, Dogme 95."
        ),
        (
            "Tradition-as-aesthetic. Production country → COUNTRY_OF_ORIGIN. "
            "Audio language → AUDIO_LANGUAGE."
        ),
        (
            "If a tradition tag exists, country is misleading (Hollywood-funded HK action "
            "isn't HK by production country).",
        ),
        ("Bollywood", "Korean cinema", "French New Wave"),
        (
            "'made in France' → COUNTRY_OF_ORIGIN.",
            "'in Korean' → AUDIO_LANGUAGE.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.METADATA),
        HandlerBucket.MUTEX,
    )
    FILMING_LOCATION = (
        "Filming location",
        (
            "Where a movie was physically shot — 'filmed in New Zealand', "
            "'shot on location in Iceland', 'Morocco shoots'."
        ),
        (
            "Filming geography only. Production country → COUNTRY_OF_ORIGIN. "
            "Narrative setting ('set in Tokyo') → NARRATIVE_SETTING."
        ),
        (
            "Production country is the wrong fit — Dune (US production) was "
            "shot in Jordan/UAE; Mission Impossible Fallout shot across "
            "Kashmir/UAE/NZ.",
        ),
        ("shot in Iceland", "filmed in Morocco", "on location in Vietnam"),
        (
            "'set in Tokyo' → NARRATIVE_SETTING.",
            "'American films' → COUNTRY_OF_ORIGIN.",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    FORMAT_VISUAL = (
        "Format + visual-format specifics",
        (
            "Format and visual-format specifics — documentary, anime, "
            "mockumentary, B&W, 70mm, IMAX, found-footage, widescreen, handheld, "
            "single-take long shot, stop-motion."
        ),
        (
            "Form the movie takes, descriptively. Acclaim framing ('visually "
            "stunning', '70mm masterpiece') → VISUAL_CRAFT_ACCLAIM. Genre → GENRE. "
            "Non-default media type ('TV movies') → MEDIA_TYPE."
        ),
        (
            "Surface form decides between this and VISUAL_CRAFT_ACCLAIM: 'shot on 70mm' "
            "(descriptive) here vs 'IMAX experience' (acclaim) → VISUAL_CRAFT_ACCLAIM.",
        ),
        ("documentary", "anime", "found-footage"),
        (
            "'visually stunning' → VISUAL_CRAFT_ACCLAIM.",
            "'horror' → GENRE.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.TIERED,
    )
    NARRATIVE_DEVICES = (
        "Narrative devices + structural form + how-told craft",
        (
            "How the story is structured/told at the craft level — plot twist, "
            "nonlinear timeline, unreliable narrator, single-location, anthology, "
            "ensemble, two-hander, POV mechanics, character-vs-plot focus."
        ),
        (
            "Structural devices only. Pacing-as-experience ('slow burn', "
            "'frenetic') → EMOTIONAL_EXPERIENTIAL. Structural ending types "
            "('twist ending', 'happy ending', 'downer ending') ALSO → "
            "EMOTIONAL_EXPERIENTIAL despite seeming structural. Dialogue acclaim "
            "→ DIALOGUE_CRAFT_ACCLAIM."
        ),
        (
            "'Plot twist' (mid-story device) here vs 'twist ending' (ending type) → "
            "EMOTIONAL_EXPERIENTIAL.",
            "'Sorkin-style' as a structural rapid-fire pattern → here; "
            "as praise for the writing → DIALOGUE_CRAFT_ACCLAIM. The "
            "surface framing decides which single category it lands in.",
        ),
        ("nonlinear timeline", "anthology film", "single-location thriller"),
        (
            "'slow burn' → EMOTIONAL_EXPERIENTIAL.",
            "'twist ending' → EMOTIONAL_EXPERIENTIAL.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.TIERED,
    )
    TARGET_AUDIENCE = (
        "Target audience",
        (
            "The audience being pitched to — 'family movies', 'teen movies', "
            "'kids movie', 'for adults', 'watch with the grandparents', "
            "'something for grown-ups'."
        ),
        (
            "Packaged-audience framing only. Story archetype like coming-of-age "
            "→ STORY_THEMATIC_ARCHETYPE. Content-sensitivity ('no gore') → "
            "SENSITIVE_CONTENT. Concrete situation ('date night') → VIEWING_OCCASION."
        ),
        (
            "An implicit rating ceiling (e.g. 'family' implying PG) is NOT "
            "a separate trait — only an explicitly named rating fires "
            "MATURITY_RATING.",
            "Imperative-mood 'watch with X' → VIEWING_OCCASION; attribute-mood 'X movies' here.",
        ),
        ("family-friendly", "teen movies", "for adults"),
        (
            "'coming-of-age' → STORY_THEMATIC_ARCHETYPE.",
            "'date night' → VIEWING_OCCASION.",
            "'no gore' → SENSITIVE_CONTENT.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.METADATA, EndpointRoute.SEMANTIC),
        HandlerBucket.COMBO,
    )
    SENSITIVE_CONTENT = (
        "Sensitive content",
        (
            "Content presence/absence and intensity — 'no gore', 'not too "
            "bloody', 'with nudity', 'violent but not graphic', 'no animal "
            "death', 'mild language only'."
        ),
        (
            "Content-on-its-own-spectrum. Audience pitch → TARGET_AUDIENCE. "
            "Explicit rating mention → MATURITY_RATING. An implied ceiling "
            "(no explicit rating) is NOT a separate trait."
        ),
        (
            "Negative polarity ('no gore', 'not too bloody') is common; the "
            "trait still fires here regardless of presence/absence framing.",
        ),
        ("no gore", "not too bloody", "no animal harm"),
        (
            "'family-friendly' → TARGET_AUDIENCE.",
            "'PG-13 only' → MATURITY_RATING.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.METADATA, EndpointRoute.SEMANTIC),
        HandlerBucket.COMBO,
    )
    SEASONAL_HOLIDAY = (
        "Seasonal / holiday",
        (
            "Seasonal/holiday framings — Christmas, Halloween, Thanksgiving, "
            "Fourth of July."
        ),
        (
            "Covers both 'movie for watching AT this season' and 'movie SET at "
            "this season'. Pure narrative setting on a date ('set on Christmas "
            "Eve') without seasonal-viewing framing → NARRATIVE_SETTING."
        ),
        (
            "'Christmas action movie' is two traits: 'Christmas' → here + "
            "'action' → GENRE.",
        ),
        ("Christmas movies", "Halloween viewing", "Thanksgiving family viewing"),
        (
            "'winter setting' (no holiday framing) → NARRATIVE_SETTING.",
            "'documentary about Christmas' → CENTRAL_TOPIC (Christmas is the topic, not the watch occasion).",
            "'snowed-in plot' → PLOT_EVENTS (story event, not seasonal viewing).",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.COMBO,
    )

    # -----------------------------------------------------------------
    # Semantic-driven
    # -----------------------------------------------------------------

    PLOT_EVENTS = (
        "Plot events",
        (
            "Literal plot events as transcript-style prose — 'a heist crew "
            "unravels when a member betrays them', 'a man wakes up with no "
            "memory and tries to find his wife's killer', 'stranded on an "
            "island after a plane crash'."
        ),
        (
            "Multi-clause event prose. Bare element nouns ('heist', 'zombie') → "
            "ELEMENT_PRESENCE. Time/place setting → NARRATIVE_SETTING. Thematic "
            "essence → STORY_THEMATIC_ARCHETYPE."
        ),
        (
            "Threshold is descriptive depth: full event description here; bare noun → "
            "ELEMENT_PRESENCE.",
        ),
        (
            "a heist crew gets double-crossed",
            "a man hunts down his wife's killer",
            "stranded after a plane crash",
        ),
        (
            "'heist movies' → ELEMENT_PRESENCE.",
            "'set in Tokyo' → NARRATIVE_SETTING.",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    NARRATIVE_SETTING = (
        "Narrative setting (time/place)",
        (
            "Story's narrative time and place — 'set in 1940s Berlin', 'during "
            "the Cold War', 'takes place in Tokyo', 'in a small desert town', "
            "'on a remote island', 'medieval Europe'."
        ),
        (
            "'Set in / takes place in / during' framings. Production era ('90s "
            "movies') → RELEASE_DATE. Filming geography → FILMING_LOCATION. "
            "Concrete focal subject ('about WWII') → CENTRAL_TOPIC."
        ),
        (
            "Setting prose lives near plot prose semantically; the routing "
            "label is what distinguishes them — keep settings tagged here, "
            "not as plot events.",
        ),
        ("set in feudal Japan", "takes place in space", "during the Cold War"),
        (
            "'90s movies' → RELEASE_DATE.",
            "'about JFK' → CENTRAL_TOPIC.",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    STORY_THEMATIC_ARCHETYPE = (
        "Story / thematic archetype",
        (
            "Story shape and thematic essence — 'about grief', 'redemption "
            "arcs', 'man-vs-nature', 'underdog stories', 'revenge', "
            "'post-apocalyptic', 'coming-of-age', 'found-family', 'man-vs-self'."
        ),
        (
            "Thematic abstraction. Concrete focal subject (JFK, Titanic) → "
            "CENTRAL_TOPIC. Static character types (anti-hero, femme fatale) → "
            "CHARACTER_ARCHETYPE."
        ),
        (
            "Spectrum framings ('kind of about grief', 'leans redemptive') still "
            "fire here as a single trait with weakened intensity — no separate "
            "routing branch.",
            "'Post-apocalyptic' is story shape here, not a genre.",
        ),
        ("redemption arc", "found family", "underdog stories"),
        (
            "'anti-hero' → CHARACTER_ARCHETYPE.",
            "'about JFK' → CENTRAL_TOPIC.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.TIERED,
    )
    EMOTIONAL_EXPERIENTIAL = (
        "Emotional / experiential",
        (
            "All emotional/experiential framings — tone (dark, whimsical, "
            "gritty, cozy, melancholic), cognitive demand (mindless vs "
            "cerebral), pacing-as-experience ('slow burn', 'frenetic'), "
            "self-experience goals ('make me cry', 'cheer me up'), "
            "comfort-watch ('feel-better movie', 'good first anime'), "
            "post-viewing resonance ('haunting', 'gut-punch ending', "
            "'forgettable'), structural ending types ('happy ending', "
            "'twist ending', 'downer ending', 'ambiguous ending')."
        ),
        (
            "Anything emotional/experiential — before, during, or after watching. "
            "Concrete viewing SITUATIONS ('date night') → VIEWING_OCCASION (named "
            "events, not feelings). Genre identity → GENRE. Mid-story structural "
            "devices → NARRATIVE_DEVICES."
        ),
        (
            "Structural ending types ('twist ending', 'happy ending', 'downer "
            "ending') live HERE despite being structural — the emotional "
            "weight is what defines them.",
            "'Dark action' is two traits: 'action' → GENRE + 'dark' → here.",
        ),
        ("slow burn", "make me cry", "haunting", "twist ending"),
        (
            "'date night' → VIEWING_OCCASION.",
            "'horror' → GENRE.",
            "'nonlinear timeline' → NARRATIVE_DEVICES.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.COMBO,
    )
    VIEWING_OCCASION = (
        "Viewing occasion",
        (
            "Concrete named viewing situations — 'date night', 'rainy Sunday', "
            "'long flight', 'with kids on Saturday', 'background watching', "
            "'family movie night', 'put on while cooking'."
        ),
        (
            "Named-event surface form. Feelings/states ('comfort watch', "
            "'feel-good') → EMOTIONAL_EXPERIENTIAL. Audience pitch ('family "
            "movies') → TARGET_AUDIENCE."
        ),
        (
            "Imperative-mood 'watch with X' = this; attribute-mood 'X movies' = "
            "TARGET_AUDIENCE.",
        ),
        ("date night", "rainy Sunday", "long flight"),
        (
            "'comfort watch' → EMOTIONAL_EXPERIENTIAL.",
            "'family movies' → TARGET_AUDIENCE.",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
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
            "BELOW_THE_LINE_CREATOR. Descriptive format ('shot in B&W', 'shot on "
            "70mm' without praise) → FORMAT_VISUAL."
        ),
        (
            "'70mm masterpiece' is two traits: 'masterpiece' → here + "
            "'70mm' → FORMAT_VISUAL.",
        ),
        ("visually stunning", "killer cinematography", "practical effects"),
        (
            "'Roger Deakins' → BELOW_THE_LINE_CREATOR.",
            "'shot in B&W' → FORMAT_VISUAL.",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    MUSIC_SCORE_ACCLAIM = (
        "Music / score acclaim",
        (
            "Acclaim for the music side — 'iconic score', 'great soundtrack', "
            "'memorable theme', 'amazing music', 'killer needle drops'."
        ),
        (
            "Acclaim only. Named composer (John Williams, Hans Zimmer) is "
            "indexed → PERSON_CREDIT. Genre 'musical' → GENRE."
        ),
        (
            "'The iconic John Williams score' is two traits: 'John "
            "Williams' → PERSON_CREDIT + 'iconic score' → here.",
        ),
        ("iconic score", "memorable theme music", "killer needle drops"),
        (
            "'John Williams' → PERSON_CREDIT.",
            "'musicals' → GENRE.",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    DIALOGUE_CRAFT_ACCLAIM = (
        "Dialogue craft acclaim",
        (
            "Acclaim for dialogue — 'quotable dialogue', 'Sorkin-style', "
            "'naturalistic dialogue', 'snappy banter', 'razor-sharp writing'."
        ),
        (
            "Dialogue acclaim only. Structural rapid-fire pattern (as a "
            "how-told device) → NARRATIVE_DEVICES. Writer credit → PERSON_CREDIT."
        ),
        (
            "'Sorkin-style' as praise for the writing → here; as a "
            "structural rapid-fire pattern → NARRATIVE_DEVICES. The "
            "surface framing decides which single category it lands in.",
        ),
        ("snappy dialogue", "quotable lines", "Sorkin-style"),
        (
            "'Aaron Sorkin movies' → PERSON_CREDIT.",
            "'rapid-fire structure' → NARRATIVE_DEVICES.",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    GENERAL_APPEAL = (
        "General appeal / quality baseline",
        (
            "Qualitative quality without numeric threshold — 'well-received', "
            "'highly rated', 'popular', 'best', 'great', 'highly regarded', "
            "'crowd-pleaser'."
        ),
        (
            "Qualitative quality without numeric threshold. Specific numeric "
            "thresholds → NUMERIC_RECEPTION_SCORE. Cultural / canonical status "
            "('classic', 'cult', 'underrated', 'era-defining') → CULTURAL_STATUS. "
            "Specific aspect praise/criticism ('praised for X', 'criticized "
            "for pacing') → SPECIFIC_PRAISE_CRITICISM. Live trending → TRENDING."
        ),
        (
            "'best horror of the 80s' is three traits: 'best' → here + "
            "'horror' → GENRE + 'of the 80s' → RELEASE_DATE.",
            "'classic' by itself → CULTURAL_STATUS; add a separate "
            "RELEASE_DATE trait only when an explicit era word is present "
            "('old classic', 'modern classic').",
        ),
        ("well-received", "highly regarded", "popular"),
        (
            "'rated above 8' → NUMERIC_RECEPTION_SCORE.",
            "'classic' → CULTURAL_STATUS.",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    CULTURAL_STATUS = (
        "Cultural status / canonical stature",
        (
            "Broad reputation, canon, and reception-shape labels — 'classic', "
            "'cult classic', 'underrated', 'overhyped', 'divisive', "
            "'era-defining', 'still holds up', 'influential', 'iconic', "
            "'landmark', 'culturally significant', 'ahead of its time'."
        ),
        (
            "Whole-work cultural position, not a specific part people liked "
            "or disliked. Specific aspect praise/criticism ('praised for "
            "tension', 'criticized for pacing') → SPECIFIC_PRAISE_CRITICISM. "
            "General quality as a numeric prior ('well-received', 'popular') "
            "→ GENERAL_APPEAL. Formal awards → AWARDS. Named curated lists "
            "('Criterion Collection', 'AFI Top 100') → GENERIC_CATCHALL."
        ),
        (
            "'classic' by itself lives here as canonical stature; add "
            "RELEASE_DATE only when an explicit era word is present "
            "('old classic', 'modern classic').",
            "'underrated' primarily needs semantic reception/status prose; "
            "metadata is optional and must not be treated as a simple "
            "well-received floor.",
            "'cult', 'divisive', and 'overhyped' describe reception shape, "
            "not specific praised or criticized qualities.",
        ),
        ("classic", "cult classic", "underrated", "still holds up", "era-defining"),
        (
            "'praised for tension' → SPECIFIC_PRAISE_CRITICISM.",
            "'rated above 8' → NUMERIC_RECEPTION_SCORE.",
            "'Oscar-winning' → AWARDS.",
            "'Criterion Collection' → GENERIC_CATCHALL.",
        ),
        (EndpointRoute.SEMANTIC, EndpointRoute.METADATA),
        HandlerBucket.COMBO,
    )
    SPECIFIC_PRAISE_CRITICISM = (
        "Specific praise / criticism",
        (
            "Reception prose for specific parts, qualities, or aspects people "
            "liked or disliked — 'praised for tension', 'criticized as "
            "plodding', 'praised for performances', 'criticized for weak "
            "ending', 'loved for its dialogue', 'hated for pacing'."
        ),
        (
            "Aspect-level praise/criticism only. Broad cultural status "
            "('classic', 'cult', 'underrated', 'divisive', 'era-defining', "
            "'still holds up') → CULTURAL_STATUS. Numeric prior → "
            "GENERAL_APPEAL. Formal awards ('Oscar-winning', "
            "'BAFTA-nominated') → AWARDS."
        ),
        (
            "Ask 'what other trait does this qualify?' If the answer is a "
            "specific aspect of the movie (pacing, tension, performances, "
            "ending, script), it can live here. If the answer is only the "
            "whole movie's place in the culture, route to CULTURAL_STATUS.",
        ),
        (
            "praised for tension",
            "criticized as plodding",
            "praised for performances",
            "criticized for weak ending",
        ),
        (
            "'classic' → CULTURAL_STATUS.",
            "'cult classic' → CULTURAL_STATUS.",
            "'underrated' → CULTURAL_STATUS.",
            "'highly rated' → GENERAL_APPEAL.",
            "'Oscar-winning' → AWARDS.",
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.COMBO,
    )

    # -----------------------------------------------------------------
    # Trick / specialized
    # -----------------------------------------------------------------

    BELOW_THE_LINE_CREATOR = (
        "Below-the-line creator",
        (
            "Non-indexed creator roles — cinematographer, editor, production "
            "designer, costume designer, VFX supervisor. 'Roger Deakins', "
            "'Thelma Schoonmaker-edited', 'Sandy Powell costumes', 'Colleen "
            "Atwood designs'."
        ),
        (
            "RESERVED — currently returns empty until backing data lands. "
            "Routing here keeps these names from misrouting to PERSON_CREDIT "
            "(which would silently return zero results for non-indexed roles)."
        ),
        (
            "Director, writer, actor, producer, composer remain at PERSON_CREDIT (indexed).",
        ),
        ("Roger Deakins", "Sandy Powell costumes", "Greig Fraser-shot"),
        (
            "'Christopher Nolan' → PERSON_CREDIT (director, indexed).",
            "'great cinematography' (no name) → VISUAL_CRAFT_ACCLAIM.",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    NAMED_SOURCE_CREATOR = (
        "Named source creator",
        (
            "Named creator of source material being adapted — Stephen King, "
            "Tolkien, Shakespeare, Philip K. Dick, Neil Gaiman, Jane Austen."
        ),
        (
            "Source-material authors only. A character-anchored franchise "
            "named as the source ('Sherlock Holmes books') splits — the "
            "character → CHARACTER_FRANCHISE, the medium → ADAPTATION_SOURCE; "
            "Sherlock himself is the book's subject, not its author. A "
            "character-less franchise named as the source ('Star Wars "
            "novelizations') → FRANCHISE_LINEAGE for the franchise + "
            "ADAPTATION_SOURCE for the medium. Generic medium with no named "
            "creator → ADAPTATION_SOURCE alone. Film credit (director, "
            "writer, actor) → PERSON_CREDIT."
        ),
        (
            "'X's <medium>' phrases split into two traits: the creator → "
            "here, the medium ('books', 'plays', 'novels') → "
            "ADAPTATION_SOURCE.",
            "A named referent stays as one trait — never split mid-name "
            "(e.g. 'Stephen King' is one trait, not 'Stephen' + 'King').",
        ),
        ("Stephen King novels", "Shakespeare adaptations", "Neil Gaiman"),
        (
            "'Sherlock Holmes books' is two traits: 'Sherlock Holmes' "
            "(book's subject, not its author) → CHARACTER_FRANCHISE + "
            "'books' → ADAPTATION_SOURCE.",
            "'Christopher Nolan' → PERSON_CREDIT (director, not source author).",
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    LIKE_MEDIA_REFERENCE = (
        "Like <media> reference",
        (
            "Named-work comparison queries that require an initial "
            "outside-knowledge expansion — 'like Inception', "
            "'similar to The Office', 'movies that feel like David Lynch', "
            "'in the vein of Hitchcock thrillers', 'X-style', 'Coen Brothers "
            "movie'. This category signals that ordinary endpoint routing is "
            "not enough yet: the named referent must be expanded into "
            "distinctive traits, then re-routed."
        ),
        (
            "Triggers on EXPLICIT comparison surface forms only ('like X', "
            "'similar to Y', 'in the vein of', 'X-style', 'feels like'). Vague "
            "reference classes WITHOUT a named comparison target ('comedians "
            "doing drama', 'auteur directors of the 70s') → GENERIC_CATCHALL."
        ),
        (
            "Use outside knowledge for an initial expansion before filling "
            "ordinary endpoint parameters. The named referent's distinctive "
            "traits are what get matched, not the referent name itself — so "
            "'like Inception' returns movies sharing Inception's traits, not "
            "literal substring matches.",
        ),
        (
            "like Inception",
            "similar to The Office",
            "in the vein of Hitchcock",
            "feels like a Stephen King novel",
            "like playing The Last of Us",
        ),
        (
            "'comedians doing drama' → GENERIC_CATCHALL.",
            "'Christopher Nolan movies' → PERSON_CREDIT (direct credit, not comparison).",
            "'based on a Stephen King novel' is two traits: 'Stephen King' "
            "→ NAMED_SOURCE_CREATOR + 'novel' → ADAPTATION_SOURCE (an "
            "adaptation flag, not a stylistic comparison).",
        ),
        (
            EndpointRoute.ENTITY,
            EndpointRoute.KEYWORD,
            EndpointRoute.SEMANTIC,
            EndpointRoute.METADATA,
            EndpointRoute.FRANCHISE_STRUCTURE,
            EndpointRoute.AWARDS,
        ),
        HandlerBucket.COMBO,
    )

    # -----------------------------------------------------------------
    # Ordinal selection
    # -----------------------------------------------------------------

    CHRONOLOGICAL = (
        "Chronological ordinal",
        (
            "Release-date ordinal position — 'first', 'last', 'earliest', "
            "'latest', 'most recent', 'the newest one', 'the oldest one'."
        ),
        (
            "Ordinal position only. Range/decay framings ('90s', 'recent', "
            "'before 2000') → RELEASE_DATE. Quality superlatives split: "
            "'best' → GENERAL_APPEAL; 'most acclaimed' → CULTURAL_STATUS."
        ),
        (
            "'The latest Scorsese' is two traits: 'latest' → here + "
            "'Scorsese' → PERSON_CREDIT.",
        ),
        ("the newest one", "earliest film", "most recent release"),
        (
            "'recent movies' → RELEASE_DATE.",
            "'best of all time' is two traits: 'best' → GENERAL_APPEAL + "
            "'of all time' (canonical reception) → CULTURAL_STATUS.",
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )

    # -----------------------------------------------------------------
    # Catch-all
    # -----------------------------------------------------------------

    GENERIC_CATCHALL = (
        "Generic parametric catch-all",
        (
            "Vague reference classes and named curated lists that require an "
            "initial outside-knowledge expansion — "
            "'comedians doing drama', 'auteur directors of the 70s', "
            "'directors known for long takes', 'child actors who became "
            "serious', Criterion Collection, AFI Top 100, IMDb Top 250, BFI, "
            "National Film Registry, Sight & Sound greatest, '1001 Movies to "
            "See Before You Die'. This category signals that the trait needs "
            "expansion or curated-list knowledge before ordinary endpoint "
            "routing can happen."
        ),
        (
            "Last resort — used only when no structured category fits. "
            "Named WORK comparison ('like Inception', 'Hitchcock-style') → "
            "LIKE_MEDIA_REFERENCE — that expands a named work; this expands "
            "a class or list."
        ),
        (
            "Use outside knowledge for an initial expansion before filling "
            "ordinary endpoint parameters. The vague class or named list is "
            "what gets expanded — its distinctive traits or curated members "
            "drive retrieval, not a literal substring match on the trait "
            "wording.",
        ),
        ("Criterion Collection", "comedians doing drama", "AFI Top 100"),
        (
            "'like Inception' → LIKE_MEDIA_REFERENCE.",
            "'horror' → GENRE (clean structured fit).",
        ),
        (
            EndpointRoute.ENTITY,
            EndpointRoute.KEYWORD,
            EndpointRoute.SEMANTIC,
            EndpointRoute.METADATA,
            EndpointRoute.FRANCHISE_STRUCTURE,
            EndpointRoute.AWARDS,
        ),
        HandlerBucket.COMBO,
    )
