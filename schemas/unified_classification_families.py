"""Prompt-rendering helper for the stage-3 keyword translator.

The keyword endpoint's LLM picks members from the closed
UnifiedClassification registry (259 names across keyword,
source-material, and concept-tag enums). Picking by name alone is
under-specified — the LLM needs each member's definition AND a
family grouping so near-collision picks can be made by comparing
definitional scope inside the right family.

This module owns the 21-family layout that organizes the registry
into canonical concept groups, plus a renderer that emits the layout
+ definitions as a string suitable for direct inclusion in a system
prompt via the {{CLASSIFICATION_REGISTRY}} placeholder.

Structural mirror of schemas/production_brand_surface_forms.py and
schemas/award_surface_forms.py — same "render function returns a
string to embed in the prompt" pattern. The family layout used to
live in the deprecated search_v2/endpoint_fetching/
keyword_query_generation.py; it is hoisted here so the live
category-handler prompt path can substitute the placeholder at
import time without depending on the dead module.

Three consistency rules are enforced at render time so a stale
registry can never silently ship:
  1. No member listed in more than one family.
  2. Every listed member name resolves to a CLASSIFICATION_ENTRIES
     entry.
  3. Every CLASSIFICATION_ENTRIES entry appears in exactly one
     family — adding a new keyword/source-material/concept-tag
     enum member without slotting it into a family fails loudly.
"""

from __future__ import annotations

from schemas.unified_classification import (
    CLASSIFICATION_ENTRIES,
    UnifiedClassification,
    entry_for,
)


# 21 canonical concept families covering every UnifiedClassification
# member. Each entry is (family_header, ordered_member_names). Member
# names must match registry keys exactly (the enum NAME, not the
# display label). The render function asserts coverage + uniqueness
# against CLASSIFICATION_ENTRIES, so adding a new registry member
# without placing it in a family fails at the first prompt build.
_FAMILIES: list[tuple[str, list[str]]] = [
    (
        "1. Action / Combat / Heroics",
        [
            "ACTION", "ACTION_EPIC", "B_ACTION", "CAR_ACTION", "GUN_FU",
            "KUNG_FU", "MARTIAL_ARTS", "ONE_PERSON_ARMY_ACTION", "SAMURAI",
            "SUPERHERO", "SWORD_AND_SANDAL", "WUXIA",
        ],
    ),
    (
        "2. Adventure / Journey / Survival",
        [
            "ADVENTURE", "ADVENTURE_EPIC", "ANIMAL_ADVENTURE",
            "DESERT_ADVENTURE", "DINOSAUR_ADVENTURE", "DISASTER",
            "GLOBETROTTING_ADVENTURE", "JUNGLE_ADVENTURE",
            "MOUNTAIN_ADVENTURE", "QUEST", "ROAD_TRIP", "SEA_ADVENTURE",
            "SURVIVAL", "SWASHBUCKLER", "URBAN_ADVENTURE",
        ],
    ),
    (
        "3. Crime / Mystery / Suspense / Espionage",
        [
            "BUDDY_COP", "BUMBLING_DETECTIVE", "CAPER", "CONSPIRACY_THRILLER",
            "COZY_MYSTERY", "CRIME", "CYBER_THRILLER", "DRUG_CRIME",
            "EROTIC_THRILLER", "FILM_NOIR", "GANGSTER",
            "HARD_BOILED_DETECTIVE", "HEIST", "LEGAL_THRILLER", "MYSTERY",
            "POLICE_PROCEDURAL", "POLITICAL_THRILLER",
            "PSYCHOLOGICAL_THRILLER", "SERIAL_KILLER", "SPY",
            "SUSPENSE_MYSTERY", "THRILLER", "WHODUNNIT",
        ],
    ),
    (
        "4. Comedy / Satire / Comic Tone",
        [
            "BODY_SWAP_COMEDY", "BUDDY_COMEDY", "COMEDY", "DARK_COMEDY",
            "FARCE", "HIGH_CONCEPT_COMEDY", "PARODY", "QUIRKY_COMEDY",
            "RAUNCHY_COMEDY", "ROMANTIC_COMEDY", "SATIRE",
            "SCREWBALL_COMEDY", "SLAPSTICK", "STONER_COMEDY",
        ],
    ),
    (
        "5. Drama / History / Institutions",
        [
            "COP_DRAMA", "COSTUME_DRAMA", "DRAMA", "EPIC", "FINANCIAL_DRAMA",
            "HISTORICAL_EPIC", "HISTORY", "LEGAL_DRAMA", "MEDICAL_DRAMA",
            "PERIOD_DRAMA", "POLITICAL_DRAMA", "PRISON_DRAMA",
            "PSYCHOLOGICAL_DRAMA", "SHOWBIZ_DRAMA", "TRAGEDY",
            "WORKPLACE_DRAMA",
        ],
    ),
    (
        "6. Horror / Macabre / Creature",
        [
            "B_HORROR", "BODY_HORROR", "FOLK_HORROR", "FOUND_FOOTAGE_HORROR",
            "GIALLO", "HORROR", "MONSTER_HORROR", "PSYCHOLOGICAL_HORROR",
            "SLASHER_HORROR", "SPLATTER_HORROR", "SUPERNATURAL_HORROR",
            "VAMPIRE_HORROR", "WEREWOLF_HORROR", "WITCH_HORROR",
            "ZOMBIE_HORROR",
        ],
    ),
    (
        "7. Fantasy / Sci-Fi / Speculative",
        [
            "ALIEN_INVASION", "ARTIFICIAL_INTELLIGENCE", "CYBERPUNK",
            "DARK_FANTASY", "DYSTOPIAN_SCI_FI", "FAIRY_TALE", "FANTASY",
            "FANTASY_EPIC", "KAIJU", "MECHA", "SCI_FI", "SCI_FI_EPIC",
            "SPACE_SCI_FI", "STEAMPUNK", "SUPERNATURAL_FANTASY",
            "SWORD_AND_SORCERY", "TIME_TRAVEL",
        ],
    ),
    (
        "8. Romance / Relationship",
        [
            "DARK_ROMANCE", "FEEL_GOOD_ROMANCE", "ROMANCE", "ROMANTIC_EPIC",
            "STEAMY_ROMANCE", "TRAGIC_ROMANCE",
        ],
    ),
    (
        "9. War / Western / Frontier",
        [
            "WAR", "WAR_EPIC", "WESTERN", "CLASSICAL_WESTERN",
            "CONTEMPORARY_WESTERN", "SPAGHETTI_WESTERN", "WESTERN_EPIC",
        ],
    ),
    (
        "10. Music / Musical / Performance",
        [
            "CLASSIC_MUSICAL", "CONCERT", "JUKEBOX_MUSICAL", "MUSIC",
            "MUSICAL", "POP_MUSICAL", "ROCK_MUSICAL",
        ],
    ),
    (
        "11. Sports / Competitive Activity",
        [
            "BASEBALL", "BASKETBALL", "BOXING", "EXTREME_SPORT", "FOOTBALL",
            "MOTORSPORT", "SOCCER", "SPORT", "WATER_SPORT",
        ],
    ),
    (
        "12. Audience / Age / Life Stage",
        [
            "FAMILY", "COMING_OF_AGE", "TEEN_ADVENTURE", "TEEN_COMEDY",
            "TEEN_DRAMA", "TEEN_FANTASY", "TEEN_HORROR", "TEEN_ROMANCE",
        ],
    ),
    (
        "13. Animation / Live Action / Anime Form / Technique",
        [
            "ADULT_ANIMATION", "ANIMATION", "ANIME", "COMPUTER_ANIMATION",
            "HAND_DRAWN_ANIMATION", "ISEKAI", "IYASHIKEI", "JOSEI",
            "LIVE_ACTION", "SEINEN", "SHOJO", "SHONEN", "SLICE_OF_LIFE",
            "STOP_MOTION_ANIMATION",
        ],
    ),
    (
        "14. Seasonal / Holiday",
        [
            "HOLIDAY", "HOLIDAY_ANIMATION", "HOLIDAY_COMEDY",
            "HOLIDAY_FAMILY", "HOLIDAY_ROMANCE",
        ],
    ),
    (
        "15. Nonfiction / Documentary / Real-World Media",
        [
            "CRIME_DOCUMENTARY", "DOCUDRAMA", "DOCUMENTARY",
            "FAITH_AND_SPIRITUALITY_DOCUMENTARY", "FOOD_DOCUMENTARY",
            "HISTORY_DOCUMENTARY", "MILITARY_DOCUMENTARY",
            "MUSIC_DOCUMENTARY", "NATURE_DOCUMENTARY", "NEWS",
            "POLITICAL_DOCUMENTARY",
            "SCIENCE_AND_TECHNOLOGY_DOCUMENTARY", "SPORTS_DOCUMENTARY",
            "TRAVEL_DOCUMENTARY", "TRUE_CRIME",
        ],
    ),
    (
        "16. Program / Presentation / Form Factor",
        [
            "BUSINESS_REALITY_TV", "COOKING_COMPETITION", "GAME_SHOW",
            "MOCKUMENTARY", "PARANORMAL_REALITY_TV", "REALITY_TV", "SHORT",
            "SITCOM", "SKETCH_COMEDY", "SOAP_OPERA", "STAND_UP", "TALK_SHOW",
        ],
    ),
    (
        "17. Cultural / National Cinema Tradition",
        [
            "ARABIC", "BENGALI", "CANTONESE", "DANISH", "DUTCH", "FILIPINO",
            "FINNISH", "FRENCH", "GERMAN", "GREEK", "HINDI", "ITALIAN",
            "JAPANESE", "KANNADA", "KOREAN", "MALAYALAM", "MANDARIN",
            "MARATHI", "NORWEGIAN", "PERSIAN", "PORTUGUESE", "PUNJABI",
            "RUSSIAN", "SPANISH", "SWEDISH", "TAMIL", "TELUGU", "THAI",
            "TURKISH", "URDU",
        ],
    ),
    (
        "18. Source Material / Adaptation / Real-World Basis",
        [
            "NOVEL_ADAPTATION", "SHORT_STORY_ADAPTATION", "STAGE_ADAPTATION",
            "TRUE_STORY", "BIOGRAPHY", "COMIC_ADAPTATION",
            "FOLKLORE_ADAPTATION", "VIDEO_GAME_ADAPTATION", "REMAKE",
            "TV_ADAPTATION",
        ],
    ),
    (
        "19. Narrative Mechanics / Endings",
        [
            "PLOT_TWIST", "TWIST_VILLAIN", "TIME_LOOP", "NONLINEAR_TIMELINE",
            "UNRELIABLE_NARRATOR", "OPEN_ENDING", "SINGLE_LOCATION",
            "BREAKING_FOURTH_WALL", "CLIFFHANGER_ENDING", "HAPPY_ENDING",
            "SAD_ENDING", "BITTERSWEET_ENDING",
        ],
    ),
    (
        "20. Story Engine / Setting / Character Archetype",
        [
            "REVENGE", "UNDERDOG", "KIDNAPPING", "CON_ARTIST",
            "POST_APOCALYPTIC", "HAUNTED_LOCATION", "SMALL_TOWN",
            "FEMALE_LEAD", "ENSEMBLE_CAST", "ANTI_HERO",
        ],
    ),
    (
        "21. Viewer Response / Content Sensitivity",
        [
            "FEEL_GOOD", "TEARJERKER", "ANIMAL_DEATH",
        ],
    ),
]


def render_classification_registry_for_prompt() -> str:
    """Render every UnifiedClassification member grouped under its
    canonical family header, with definitions inline.

    Format:
        <family_header>

        <NAME>: <definition>
        <NAME>: <definition>
        ...

        <next_family_header>
        ...

    The surrounding markdown in keyword.md provides the section
    title and the "compare definitions to disambiguate" framing, so
    this output focuses on the data — no preamble, no trailing
    separator.

    Raises at call time (which is module-import time for the prompt
    builder) when a registry member is duplicated across families,
    a listed name doesn't resolve, or a registry member is orphaned
    — see the three consistency rules in the module docstring.
    """
    listed: set[str] = set()
    blocks: list[str] = []

    for family_header, member_names in _FAMILIES:
        lines = [family_header, ""]
        for name in member_names:
            if name in listed:
                raise RuntimeError(
                    f"UnifiedClassification member {name!r} listed in more "
                    f"than one family in unified_classification_families._FAMILIES"
                )
            if name not in CLASSIFICATION_ENTRIES:
                raise RuntimeError(
                    f"UnifiedClassification member {name!r} referenced by "
                    f"unified_classification_families._FAMILIES is not in the "
                    f"registry — update _FAMILIES or the registry to stay in sync."
                )
            listed.add(name)
            entry = entry_for(UnifiedClassification(name))
            lines.append(f"{entry.name}: {entry.definition}")
        blocks.append("\n".join(lines))

    missing = set(CLASSIFICATION_ENTRIES) - listed
    if missing:
        raise RuntimeError(
            "Registry members not placed in any family in "
            "unified_classification_families._FAMILIES: "
            + ", ".join(sorted(missing))
        )

    return "\n\n".join(blocks)
