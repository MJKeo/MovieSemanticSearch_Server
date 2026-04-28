"""
Trait category taxonomy for the search v2 step-2 grounding LLM.

`CategoryName` is the canonical vocabulary the step-2 pre-pass picks
from when grounding each captured meaning of a query fragment. Each
member carries:

- A descriptive name (the string value emitted to downstream code).
- A concept-only description (what the category is about + what it is
  NOT about) injected into the step-2 system prompt. The description
  combines the *handles* and *boundary notes* from the canonical
  taxonomy in `search_improvement_planning/query_categories.md`.
- An ordered tuple of `EndpointRoute` values naming the retrieval
  endpoints the category may dispatch to. Endpoint order is priority
  order: for tiered categories the first entry is the authoritative
  tier and later entries are fallbacks; for single-endpoint or combo
  categories the order reflects the primary channel first.
- A `HandlerBucket` identifying the orchestration shape (single /
  mutex / tiered / combo).

The 44-category list and its underlying granularity principles are
documented in `search_improvement_planning/query_categories.md`. That
doc is the source of truth for handles + boundaries; this enum mirrors
those definitions in a form the LLM can pick from.
"""

from enum import Enum

from schemas.enums import EndpointRoute, HandlerBucket


class CategoryName(str, Enum):
    description: str
    endpoints: tuple["EndpointRoute", ...]
    bucket: "HandlerBucket"

    def __new__(
        cls,
        value: str,
        description: str,
        endpoints: tuple["EndpointRoute", ...],
        bucket: "HandlerBucket",
    ) -> "CategoryName":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        obj.endpoints = endpoints
        obj.bucket = bucket
        return obj

    # -----------------------------------------------------------------
    # Structured / lexical
    # -----------------------------------------------------------------

    PERSON_CREDIT = (
        "Person credit",
        (
            "Named person credits via posting tables — actor, director, "
            "writer, producer, composer. Indexed roles only. "
            "Below-the-line creators (cinematographer, editor, "
            "production designer, costume designer, VFX supervisor) "
            "do NOT belong here — they route to BELOW_THE_LINE_CREATOR. "
            "Title-based searches ('movies called X') route to "
            "TITLE_TEXT instead — this category is for proper-noun "
            "people, not movie titles."
        ),
        (EndpointRoute.ENTITY,),
        HandlerBucket.SINGLE,
    )
    TITLE_TEXT = (
        "Title text lookup",
        (
            "Title substring matches — 'any movie called Inception', "
            "'movies with Star in the title', 'titles starting with The'. "
            "Free-string match against the title column. Distinct from "
            "PERSON_CREDIT which uses role-typed posting tables; the "
            "input shape is the same (a string) but the SQL and "
            "precision behavior differ."
        ),
        (EndpointRoute.ENTITY,),
        HandlerBucket.SINGLE,
    )
    NAMED_CHARACTER = (
        "Named character",
        (
            "Presence of a specific named character in the movie — "
            "Batman, Wolverine, James Bond, Hermione, Spider-Man, "
            "Harry Potter. Must be a specific named persona. Step 2 "
            "ALWAYS decomposes phrases like 'Batman movies' into a "
            "named-character trait (here) plus a separate franchise "
            "trait (FRANCHISE_LINEAGE) — this category never fans out "
            "to franchise on its own. Character types and patterns "
            "(lovable rogue, femme fatale) belong to "
            "CHARACTER_ARCHETYPE, not here."
        ),
        (EndpointRoute.ENTITY,),
        HandlerBucket.SINGLE,
    )
    STUDIO_BRAND = (
        "Studio / brand",
        (
            "Production studios and curated production brands — "
            "Disney, Pixar, A24, Studio Ghibli, Blumhouse, Marvel "
            "Studios, Dreamworks, Hammer Films. The entity that made "
            "the movie, not its franchise or intellectual property "
            "(franchise routes to FRANCHISE_LINEAGE)."
        ),
        (EndpointRoute.STUDIO,),
        HandlerBucket.SINGLE,
    )
    FRANCHISE_LINEAGE = (
        "Franchise / universe lineage",
        (
            "Membership in a named franchise or shared universe and "
            "positioning within it — sequels, prequels, spinoffs, "
            "reboots, remakes-as-lineage-positioning, mainline vs "
            "offshoot, crossovers, 'the original Scarface, not the "
            "remake'. About where a movie sits inside a named series. "
            "When a 'based on' phrase names a film franchise as the "
            "source ('based on Sherlock Holmes books'), the named "
            "referent stays here — only when the named referent is a "
            "creator of source material does it route to "
            "NAMED_SOURCE_CREATOR."
        ),
        (EndpointRoute.FRANCHISE_STRUCTURE,),
        HandlerBucket.SINGLE,
    )
    ADAPTATION_SOURCE = (
        "Adaptation source flag",
        (
            "Origin medium of the story as a yes/no flag — novel "
            "adaptation, comic book, true story, biography, "
            "video-game adaptation, remake (used as origin medium, not "
            "franchise positioning). About WHAT the source material's "
            "medium is, not which franchise. Composes with "
            "FRANCHISE_LINEAGE (when the named source is a franchise) "
            "or NAMED_SOURCE_CREATOR (when the named source is a "
            "creator) — both fire alongside this category when the "
            "query names a specific source."
        ),
        (EndpointRoute.KEYWORD,),
        HandlerBucket.SINGLE,
    )
    CENTRAL_TOPIC = (
        "Central topic / about-ness",
        (
            "The film's central concrete subject — the thing the movie "
            "IS ABOUT, not just contains. 'About JFK', 'Titanic movie', "
            "'Watergate', 'Princess Diana biopic', 'Vietnam War'. "
            "Concrete subjects only. Thematic essence (grief, "
            "redemption, found family) belongs to "
            "STORY_THEMATIC_ARCHETYPE — that's about story shape, not "
            "concrete subject. Mere presence ('has clowns', 'zombie "
            "movies') belongs to ELEMENT_PRESENCE — that's about "
            "presence, not centrality. Tiered: KW canonical tag "
            "(BIOGRAPHY, TRUE_STORY, historical-event tag) first; "
            "semantic prose fallback for spectrum or long-tail subjects."
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.TIERED,
    )
    ELEMENT_PRESENCE = (
        "Element / motif presence",
        (
            "Concrete element appearing in the story — 'has clowns', "
            "'zombie movies', 'shark movies', 'robots', 'movies with "
            "horses', 'vampires'. Binary 'is this thing in the story?' "
            "framing. Distinct from CENTRAL_TOPIC ('about X' is "
            "centrality, 'has X' is presence) and from "
            "CHARACTER_ARCHETYPE (lovable rogue is a character type, "
            "not an element). Tiered: KW tag (ZOMBIE, CLOWN, SHARK) "
            "first; semantic prose fallback when no tag covers the "
            "request."
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.TIERED,
    )
    CHARACTER_ARCHETYPE = (
        "Character archetype",
        (
            "Character TYPE patterns — lovable rogue, anti-hero, "
            "femme fatale, underdog protagonist, reluctant hero, "
            "manic pixie dream girl, love-to-hate villain. Static "
            "character type, not a specific named persona "
            "(NAMED_CHARACTER) and not an element-in-story "
            "(ELEMENT_PRESENCE). Distinct from STORY_THEMATIC_ARCHETYPE "
            "which captures story shape / character trajectory rather "
            "than static type. Tiered: KW ConceptTag (ANTI_HERO, "
            "FEMALE_LEAD) first; NRT characterization prose fallback."
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.TIERED,
    )
    AWARDS = (
        "Award records",
        (
            "Formal award wins and nominations — Oscar-winning, "
            "BAFTA-nominated, Palme d'Or, Golden Globe, Cannes, "
            "Sundance. Ceremony-specific filters and multi-win "
            "superlatives. Quality-superlative queries that mention "
            "awards ('Oscar-winning best picture') compose this "
            "category with GENERAL_APPEAL / SPECIFIC_PRAISE_CRITICISM "
            "per the compound split rule — never fold those into this "
            "category."
        ),
        (EndpointRoute.AWARDS,),
        HandlerBucket.SINGLE,
    )
    TRENDING = (
        "Trending",
        (
            "Live 'right now' trending status — 'what's popular this "
            "week', 'trending', 'what everyone's watching', 'currently "
            "buzzing'. Requires a current-cadence refresh signal, "
            "distinct from static popularity (GENERAL_APPEAL uses "
            "popularity_score as a static prior)."
        ),
        (EndpointRoute.TRENDING,),
        HandlerBucket.SINGLE,
    )

    # -----------------------------------------------------------------
    # Structured single-attribute (META, per attribute)
    # -----------------------------------------------------------------

    RELEASE_DATE = (
        "Release date / era",
        (
            "Date ranges, eras, and vague-language date framings — "
            "'90s', 'old', 'recent', 'before 2000', 'in the 2000s', "
            "'old-school', 'pre-millennium'. Range filter with "
            "soft-falloff for vague language; per-user defaults "
            "consulted for relative terms ('modern', 'recent'). "
            "Range/decay framings only — ordinal position ('newest', "
            "'earliest', 'the latest one') belongs to CHRONOLOGICAL."
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    RUNTIME = (
        "Runtime",
        (
            "Movie length framings — 'around 90 minutes', 'short', "
            "'long', 'under 2 hours', 'feature-length'. Range filter "
            "with soft-falloff. SYSTEM DEFAULT: a 60-minute floor is "
            "applied to all queries unless the query explicitly asks "
            "for shorts (enforced at the dispatcher level, not by this "
            "category firing)."
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    MATURITY_RATING = (
        "Maturity rating",
        (
            "Movie rating filters — 'PG-13 max', 'rated R', 'G-rated', "
            "'no NC-17'. When maturity is a packaged-audience framing "
            "('family movies') or a content-sensitivity framing "
            "('nothing too graphic'), it composes with TARGET_AUDIENCE "
            "or SENSITIVE_CONTENT — this category still fires for the "
            "rating ceiling itself, while those carry the "
            "inclusion-scoring side."
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    AUDIO_LANGUAGE = (
        "Audio language",
        (
            "Movies in a specific original audio language — "
            "'in Korean', 'Spanish-language', 'subtitled', "
            "'French-original'."
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    STREAMING = (
        "Streaming platform",
        (
            "Movies available on a specific streaming platform — "
            "'on Netflix', 'on Hulu', 'on Prime', 'streaming on Max'."
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    BUDGET_SCALE = (
        "Budget scale",
        (
            "Budget framings — 'big-budget', 'low-budget', 'indie "
            "scale', 'shoestring', 'micro-budget', 'studio-scale'."
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    BOX_OFFICE = (
        "Box office bucket",
        (
            "Box office framings — 'box office hit', 'blockbuster "
            "gross', 'flop', 'underperformer', 'sleeper hit', "
            "'made bank'."
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    NUMERIC_RECEPTION_SCORE = (
        "Numeric reception score",
        (
            "Specific numeric reception thresholds — 'rated above 8', "
            "'70%+ on RT', '5-star', 'IMDb score over 7.5'. Specific "
            "numeric framing only — qualitative quality language "
            "('well-rated', 'best', 'highly regarded') routes to "
            "GENERAL_APPEAL as a numeric prior. Same column read by "
            "both, but framing differs (threshold-filter vs additive "
            "prior)."
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    COUNTRY_OF_ORIGIN = (
        "Country of origin",
        (
            "Legal/financial production country — 'produced in', "
            "'American films', 'British production', 'Canadian "
            "co-production'. Filming geography routes to "
            "FILMING_LOCATION (which records where shooting actually "
            "happened); cultural-tradition framing routes to "
            "CULTURAL_TRADITION (which is about aesthetic, not legal "
            "origin)."
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    MEDIA_TYPE = (
        "Media type",
        (
            "Non-default media formats — 'TV movies', 'video releases', "
            "'shorts (when explicit)', 'made-for-TV'. Does NOT fire "
            "for vanilla 'show me movies' queries: the dispatcher "
            "applies a default media_type=movie plus a 60-minute "
            "runtime floor, which covers normal cases. Fires only "
            "when a non-default media type is explicitly requested."
        ),
        (EndpointRoute.MEDIA_TYPE,),
        HandlerBucket.SINGLE,
    )

    # -----------------------------------------------------------------
    # Structured / KW continuing
    # -----------------------------------------------------------------

    GENRE = (
        "Genre",
        (
            "All genre framings — top-level (horror, action, comedy, "
            "sci-fi, drama, romance, animation, thriller, fantasy, "
            "western) AND sub-genre (body horror, neo-noir, cozy "
            "mystery, space opera, slasher, giallo, slow-burn "
            "thriller). Mutually exclusive: KW genre-family or "
            "sub-genre tag if canonical; semantic genre_signatures "
            "prose for compound/qualifier-laden ('dark action', "
            "'quiet drama') or sub-genres without tags. Story "
            "archetype ('revenge', 'underdog', 'heist', "
            "'post-apocalyptic') routes to STORY_THEMATIC_ARCHETYPE "
            "instead — that names a story shape, not a genre."
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.MUTEX,
    )
    CULTURAL_TRADITION = (
        "Cultural tradition / national cinema",
        (
            "Named cinema traditions / movements — Bollywood, Korean "
            "cinema, Hong Kong action, Italian neorealism, French New "
            "Wave, Nordic noir, J-horror, Dogme 95. "
            "Tradition-as-aesthetic, not legal production country "
            "(COUNTRY_OF_ORIGIN) and not filming geography "
            "(FILMING_LOCATION). Mutually exclusive: KW tradition tag "
            "if canonical; META country/language fallback when no tag "
            "exists. If a tag exists, country is misleading "
            "(Hollywood-funded HK action isn't HK by production "
            "country)."
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.METADATA),
        HandlerBucket.MUTEX,
    )
    FILMING_LOCATION = (
        "Filming location",
        (
            "Where a movie was physically shot — 'filmed in New "
            "Zealand', 'shot on location in Iceland', 'Morocco "
            "shoots'. Production geography only. "
            "META.country_of_origin is the wrong column — it records "
            "legal/financial origin, not filming location (The "
            "Revenant in Canada/Argentina, Dune in Jordan/UAE, "
            "Mission Impossible — Fallout across Kashmir/UAE/NZ all "
            "carry US country_of_origin)."
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    FORMAT_VISUAL = (
        "Format + visual-format specifics",
        (
            "Format (documentary, anime, mockumentary) and "
            "visual-format specifics (B&W, 70mm, IMAX, found-footage, "
            "widescreen, handheld, single-take long shot, stop-motion). "
            "About the form the movie takes, not its genre. Tiered: "
            "canonical KW tag (DOCUMENTARY, BLACK_AND_WHITE, "
            "FOUND_FOOTAGE, ANIMATION) first; semantic prose fallback "
            "for technique-level long-tail without tags."
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.TIERED,
    )
    NARRATIVE_DEVICES = (
        "Narrative devices + structural form + how-told craft",
        (
            "How the story is structured / told at the craft level — "
            "plot twist, nonlinear timeline, unreliable narrator, "
            "single-location, anthology, ensemble, two-hander, POV "
            "mechanics, character-vs-plot focus, 'Sorkin-style' as a "
            "structural craft pattern. Pacing-as-EXPERIENCE ('slow "
            "burn', 'frenetic', 'methodical pace') routes to "
            "EMOTIONAL_EXPERIENTIAL — that's experiential, not "
            "structural. Tiered: canonical device tag (PLOT_TWIST, "
            "NONLINEAR_TIMELINE, UNRELIABLE_NARRATOR, "
            "SINGLE_LOCATION, ENSEMBLE_CAST) first; NRT prose for "
            "craft-level long-tail."
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.TIERED,
    )
    TARGET_AUDIENCE = (
        "Target audience",
        (
            "The audience being pitched to — 'family movies', 'teen "
            "movies', 'kids movie', 'for adults', 'watch with the "
            "grandparents', 'something for grown-ups'. "
            "Packaged-audience framing only. Story archetype like "
            "coming-of-age belongs to STORY_THEMATIC_ARCHETYPE. "
            "Content-sensitivity ('no gore') belongs to "
            "SENSITIVE_CONTENT. Concrete viewing situation ('date "
            "night', 'Saturday with the kids') belongs to "
            "VIEWING_OCCASION. Combo: META maturity gate when the "
            "framing implies a ceiling; KW audience-framing tags + "
            "CTX watch_scenarios as additive inclusion scoring."
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.METADATA, EndpointRoute.SEMANTIC),
        HandlerBucket.COMBO,
    )
    SENSITIVE_CONTENT = (
        "Sensitive content",
        (
            "Content presence/absence and acceptable intensity — "
            "'no gore', 'not too bloody', 'with nudity', 'violent "
            "but not graphic', 'no animal death', 'mild language "
            "only'. Content-on-its-own-spectrum framing; "
            "audience-pitch framing belongs to TARGET_AUDIENCE. "
            "Combo: META maturity gate when a rating ceiling is "
            "implied; KW content tags for binary presence/absence "
            "(ANIMAL_DEATH-style flags); semantic disturbance_profile "
            "for spectrum-framed asks."
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.METADATA, EndpointRoute.SEMANTIC),
        HandlerBucket.COMBO,
    )
    SEASONAL_HOLIDAY = (
        "Seasonal / holiday",
        (
            "Seasonal/holiday framing — Christmas, Halloween, "
            "Thanksgiving, Fourth of July, summer-blockbuster. Covers "
            "both 'movie for watching AT this season' and 'movie SET "
            "at this season'. Combo: KW via proxy chains (Halloween "
            "→ horror+supernatural+spooky+slasher; Christmas → "
            "family+heartwarming+winter), CTX seasonal viewing "
            "framing, and P-EVT seasonal narrative settings — no "
            "channel is authoritative on its own."
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
            "Literal plot events — 'a heist crew unravels when a "
            "member betrays them', 'a man wakes up with no memory and "
            "tries to find his wife's killer', 'stranded on an island "
            "after a plane crash'. Event prose only. Narrative "
            "time/place setting ('set in 1940s Berlin', 'takes place "
            "in Tokyo') routes to NARRATIVE_SETTING — same vector "
            "space, different query phrasing template."
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    NARRATIVE_SETTING = (
        "Narrative setting (time/place)",
        (
            "The story's narrative time and place — 'set in 1940s "
            "Berlin', 'during the Cold War', 'takes place in Tokyo', "
            "'in a small desert town', 'on a remote island', "
            "'medieval Europe'. Same P-EVT vector space as "
            "PLOT_EVENTS but with descriptive query phrasing ('set "
            "in X', 'takes place in Y') rather than transcript-style "
            "event prose. Split exists so the handler runs the right "
            "phrasing template without re-deriving."
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    STORY_THEMATIC_ARCHETYPE = (
        "Story / thematic archetype",
        (
            "Story shape and thematic essence — 'movies about grief', "
            "'redemption arcs', 'man-vs-nature', 'underdog stories', "
            "'revenge stories', 'post-apocalyptic', 'coming-of-age "
            "about self-acceptance', 'sisterly love stories', "
            "'found-family stories', 'man-vs-self'. Spectrum framings "
            "('kind of about grief', 'leans redemptive') handled via "
            "salience=supporting downstream weighting, not via "
            "routing. Distinct from CENTRAL_TOPIC by abstraction "
            "(thematic essence here, concrete subject there) and "
            "from CHARACTER_ARCHETYPE by static-vs-trajectory "
            "(character archetype is a static type, story archetype "
            "is a story shape). Tiered: ConceptTag (REDEMPTION, "
            "FOUND_FAMILY, REVENGE, CORRUPTION) first; P-ANA prose "
            "fallback across elevator_pitch, conflict_type, "
            "thematic_concepts, character_arcs."
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.TIERED,
    )
    EMOTIONAL_EXPERIENTIAL = (
        "Emotional / experiential",
        (
            "All emotional and experiential framings — before, "
            "during, AND after watching. Includes: during-viewing "
            "feel (tone, tonal aesthetic like dark/whimsical/gritty/"
            "cozy/melancholic, cognitive demand like mindless vs "
            "cerebral, realism vs stylization, tension/disturbance "
            "intensity, emotional palette); pacing-as-experience "
            "('slow burn', 'frenetic'); self-experience goals "
            "('make me cry', 'cheer me up', 'something mindless', "
            "'challenge me'); comfort-watch / gateway ('go-to "
            "movie', 'feel-better movie', 'good first anime', "
            "'accessible arthouse'); post-viewing resonance ('stays "
            "with you', 'haunting', 'gut-punch ending', "
            "'forgettable'); structural ending types ('happy "
            "ending', 'twist ending', 'downer ending', 'ambiguous "
            "ending'). Concrete viewing SITUATIONS ('date night', "
            "'rainy Sunday', 'long flight') route to "
            "VIEWING_OCCASION instead — situations are named events, "
            "distinct from feelings. The merger is deliberate: the "
            "handler-stage LLM is better at fine emotional "
            "disambiguation than the step-2 LLM. Combo: VWX, CTX, "
            "RCP semantic + KW emotional/ending tags (TEARJERKER, "
            "FEEL_GOOD, HAPPY_ENDING, TWIST_ENDING, OPEN_ENDING, "
            "SAD_ENDING)."
        ),
        (EndpointRoute.KEYWORD, EndpointRoute.SEMANTIC),
        HandlerBucket.COMBO,
    )
    VIEWING_OCCASION = (
        "Viewing occasion",
        (
            "Concrete named viewing situations — 'date night', "
            "'rainy Sunday', 'long flight', 'with kids on Saturday', "
            "'background watching', 'family movie night', 'Sunday "
            "morning', 'put on while cooking'. Carved out from "
            "EMOTIONAL_EXPERIENTIAL because situations are concrete "
            "named events, not feelings — sharply distinguishable at "
            "step 2 by the named-event surface form."
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    VISUAL_CRAFT_ACCLAIM = (
        "Visual craft acclaim",
        (
            "Acclaim attached to the visual axis — 'visually "
            "stunning', 'killer cinematography', 'beautifully shot', "
            "'IMAX-shot', 'practical effects', 'technical marvel', "
            "'gorgeous visuals'. Named cinematographer or VFX "
            "supervisor routes to BELOW_THE_LINE_CREATOR (reserved "
            "and currently empty until backing data lands)."
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    MUSIC_SCORE_ACCLAIM = (
        "Music / score acclaim",
        (
            "Acclaim for the music side — 'iconic score', 'great "
            "soundtrack', 'memorable theme', 'amazing music', "
            "'killer needle drops'. Named composer routes to "
            "PERSON_CREDIT (composer postings)."
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    DIALOGUE_CRAFT_ACCLAIM = (
        "Dialogue craft acclaim",
        (
            "Acclaim for dialogue — 'quotable dialogue', "
            "'Sorkin-style', 'naturalistic dialogue', 'snappy "
            "banter', 'razor-sharp writing'. 'Sorkin-style' can also "
            "fire NARRATIVE_DEVICES if framed as a structural "
            "pattern (rapid-fire walk-and-talk as a how-told device) "
            "rather than as praise."
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    GENERAL_APPEAL = (
        "General appeal / quality baseline",
        (
            "Qualitative quality language without a specific numeric "
            "threshold — 'well-received', 'highly rated', 'popular', "
            "'best', 'great', 'highly regarded', 'crowd-pleaser'. "
            "Numeric column priors only (reception_score + "
            "popularity_score) — additive lift, not hard threshold. "
            "Specific numeric thresholds belong to "
            "NUMERIC_RECEPTION_SCORE. Specific praise/criticism prose "
            "('praised for tension', 'criticized as overhyped') "
            "belongs to SPECIFIC_PRAISE_CRITICISM. Quality "
            "superlatives ('best horror of the 80s') decompose into "
            "this + SPECIFIC_PRAISE_CRITICISM + axis cats per the "
            "compound split rule."
        ),
        (EndpointRoute.METADATA,),
        HandlerBucket.SINGLE,
    )
    SPECIFIC_PRAISE_CRITICISM = (
        "Specific praise / criticism",
        (
            "Reception prose for what people specifically liked or "
            "disliked, plus canonical reception tags — 'cult', "
            "'underrated', 'overhyped', 'divisive', 'praised for its "
            "tension', 'criticized as plodding', 'still holds up', "
            "'era-defining', 'stacked cast', 'thematic weight' "
            "(acclaim side). Quality-as-prose, not quality-as-numeric "
            "(that's GENERAL_APPEAL). AWD records ('Oscar-winning', "
            "'BAFTA-nominated') decompose to AWARDS by compound split "
            "rule. Combo: RCP praised/criticized prose + KW "
            "(CULT_CLASSIC, UNDERRATED, DIVISIVE)."
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
            "Non-indexed creator roles — cinematographer, editor, "
            "production designer, costume designer, VFX supervisor. "
            "'Roger Deakins movies', 'Thelma Schoonmaker-edited', "
            "'Sandy Powell costumes', 'Colleen Atwood designs'. "
            "RESERVED slot — currently returns empty until postings "
            "or a directed semantic-on-credits surface lands. Routing "
            "here keeps these queries from scattering across wrong "
            "cats. Distinct from PERSON_CREDIT, which is "
            "posting-table-backed and would fail silently for "
            "non-indexed roles."
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    NAMED_SOURCE_CREATOR = (
        "Named source creator",
        (
            "Named creator of source material being adapted — "
            "'Stephen King' (in 'Stephen King novels'), 'Tolkien' "
            "(in 'Tolkien films'), 'Shakespeare' (in 'based on "
            "Shakespeare plays'), 'Philip K. Dick', 'Neil Gaiman', "
            "'Jane Austen'. The named referent stays as ONE trait — "
            "never split mid-name (e.g. 'Stephen King' is one trait, "
            "not 'Stephen' + 'King'). DETECTION RULE for 'based on / "
            "by / X's <medium>' phrases: if the referent is a film "
            "FRANCHISE (Sherlock Holmes), route to FRANCHISE_LINEAGE; "
            "if the referent is a CREATOR of source material "
            "(Shakespeare, Stephen King), route here. Step 2 always "
            "splits the source phrase: the named-referent half routes "
            "per the rule, and the medium half ('books', 'plays', "
            "'novels') routes to ADAPTATION_SOURCE."
        ),
        (EndpointRoute.SEMANTIC,),
        HandlerBucket.SINGLE,
    )
    LIKE_MEDIA_REFERENCE = (
        "Like <media> reference",
        (
            "Named-work comparison queries — 'like Inception', "
            "'similar to The Office', 'movies that feel like David "
            "Lynch', 'in the vein of Hitchcock thrillers', 'like a "
            "Coen Brothers movie', 'X-style'. Triggers on EXPLICIT "
            "comparison surface forms only. The handler interprets "
            "the named referent (extracts its distinctive traits — "
            "genre, narrative devices, tone, themes, era) and "
            "dispatches across whichever subset of the existing "
            "endpoints best captures those traits. Vague reference "
            "classes WITHOUT a named comparison target ('comedians "
            "doing drama', 'auteur directors of the 70s') route to "
            "GENERIC_CATCHALL."
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
            "Release-date ordinal position within a scoped candidate "
            "set — 'first', 'last', 'earliest', 'latest', 'most "
            "recent', 'the newest one', 'the oldest one'. "
            "Sort-and-pick by release_date. Range or decay framings "
            "('90s movies', 'recent', 'before 2000') belong to "
            "RELEASE_DATE — this category is for ordinal POSITION, "
            "not range. 'Most recent' is chronology; 'best' / 'most "
            "acclaimed' is reception superlative (GENERAL_APPEAL / "
            "SPECIFIC_PRAISE_CRITICISM). 'The latest Scorsese' fires "
            "this category plus PERSON_CREDIT."
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
            "Anything that needs interpretation/expansion and doesn't "
            "fit a structured category. Includes: vague reference "
            "classes ('comedians doing drama', 'auteur directors of "
            "the 70s', 'directors known for long takes', 'child "
            "actors who became serious'); named lists / curated "
            "canon (Criterion Collection, AFI Top 100, IMDb Top 250, "
            "BFI, National Film Registry, Sight & Sound greatest, "
            "'1001 Movies to See Before You Die'); anything else "
            "step 2 recognizes as real but underspecified. The "
            "handler interprets the trait and dispatches across "
            "whichever subset of the existing endpoints best matches "
            "— ENTITY for expanded actor/director instances, KEYWORD "
            "for tag-resolvable expansions, SEMANTIC across spaces, "
            "METADATA priors for canonical-list quality lift, "
            "FRANCHISE_STRUCTURE / AWARDS where the expansion "
            "implies them. Distinct from LIKE_MEDIA_REFERENCE which "
            "expands a named WORK; this expands a CLASS or LIST. The "
            "only true catch-all in the taxonomy — the goal is to "
            "keep shrinking it as patterns get lifted into dedicated "
            "cats."
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
