"""Curated auteur director list for the V3 similar-movies director lane.

The list is the load-bearing artifact of the V3 §2.1 director rework.
V2 used `mv_director_strength` (a popularity percentile) which conflated
fame with stylistic coherence — Lucas surfacing American Graffiti for
Star Wars, Spielberg over-firing across his genre-promiscuous
filmography, Darabont under-credited for Shawshank↔Green Mile.

V3 replaces that with a manual list of directors whose body of work
is recognizably "theirs" across films — a viewer who liked one is
likely to enjoy others. Three-criterion bar applied to compose the
list:

  1. Recognizable style across at least three features (visual,
     tonal, structural, or voice-driven).
  2. General moviegoer awareness (not deep-cut critics' choices).
  3. Other films by the director are likely to "feel the same" to a
     viewer who liked one — the recommender bar, not just "famous".

The list is stored as normalized strings (`lex.lexical_dictionary.norm_str`
form) so resolution to string_ids reuses the existing
`fetch_phrase_term_ids` path. Three credit-string corrections from the
verification pass:
  - "bong joon ho"      — Bong Joon-ho is credited without the hyphen
  - "hirokazu koreeda"  — Kore-eda is credited as one word in IMDb
  - "alejandro g inarritu" — period after the "G" collapses to "g"

The Coen brothers (and any co-directing pair) are stored as separate
entries; the lane fires for either credit independently.

See:
  - search_improvement_planning/similar_movies_v3_plan.md §2.1 (full
    list with catalog film counts and the unified scoring rule).
  - search_improvement_planning/similar_movies.md "V3 Director Lane".
"""
from __future__ import annotations

from db.postgres import fetch_phrase_term_ids


# 63 normalized director keys. Order is descriptive (grouped by
# stylistic category) and not load-bearing — frozenset is used at
# runtime for O(1) membership.
AUTEUR_NORM_STRINGS: frozenset[str] = frozenset(
    {
        # Visual stylists — frame-by-frame recognizable
        "wes anderson",
        "stanley kubrick",
        "terrence malick",
        "david fincher",
        "tim burton",
        "guillermo del toro",
        "yorgos lanthimos",
        "sofia coppola",
        "nicolas winding refn",
        "denis villeneuve",
        "edgar wright",
        "wong kar-wai",
        "pedro almodovar",
        "jean-pierre jeunet",
        "baz luhrmann",
        "zack snyder",
        # Tonal / structural auteurs
        "david lynch",
        "christopher nolan",
        "darren aronofsky",
        "charlie kaufman",
        "spike jonze",
        "m night shyamalan",
        "ari aster",
        "robert eggers",
        "jordan peele",
        "bong joon ho",        # IMDb credit lacks the hyphen
        "park chan-wook",
        # Voice / dialogue-driven
        "quentin tarantino",
        "aaron sorkin",
        "richard linklater",
        "noah baumbach",
        "joel coen",           # Coens stored as two separate keys —
        "ethan coen",          # lane fires for either credit
        # Career-spanning masters
        "martin scorsese",
        "paul thomas anderson",
        "spike lee",
        "david cronenberg",
        "michael mann",
        "brian de palma",
        "john carpenter",
        "sam raimi",
        "hayao miyazaki",
        # International auteurs
        "akira kurosawa",
        "federico fellini",
        "ingmar bergman",
        "andrei tarkovsky",
        "werner herzog",
        "hirokazu koreeda",    # Kore-eda credited as one word in IMDb
        "lars von trier",
        "michael haneke",
        "luca guadagnino",
        "celine sciamma",
        "jane campion",
        # Modern voices
        "damien chazelle",
        "barry jenkins",
        "greta gerwig",
        "steve mcqueen",
        "alfonso cuaron",
        "alejandro g inarritu",  # period after "G." collapses to "g"
        "james wan",
        # Per user direction (held back during initial composition;
        # added before V3 ship). Stylistic signal is real; inclusion
        # is a separate decision from the three-criterion bar.
        "woody allen",
        "mel gibson",
        "roman polanski",
    }
)


# Module-level cache for the resolved string_id set. Populated lazily
# on first call to `fetch_auteur_term_ids` and not invalidated for the
# process lifetime — director term_ids in `lex.lexical_dictionary` are
# append-only (string_id is a generated identity), so a once-resolved
# id stays valid. New directors added to the list require a process
# restart to be picked up; that's an acceptable tradeoff for a list
# that changes once per quarter at most.
_resolved_term_ids: frozenset[int] | None = None


async def fetch_auteur_term_ids() -> frozenset[int]:
    """Resolve the auteur normalized strings to lexical_dictionary string_ids.

    Returns the cached set on the second and subsequent calls. Names
    that don't resolve to a string_id are silently dropped — the
    verification pass at composition time confirms all 63 are present
    in the catalog (see `/tmp/verify_directors.py` log
    on 2026-05-07 and the §2.1 table). A previously-present name
    becoming unresolved would indicate a catalog-side regression and
    should be caught by the existing ingestion-side invariants.
    """
    global _resolved_term_ids
    if _resolved_term_ids is not None:
        return _resolved_term_ids

    name_to_id = await fetch_phrase_term_ids(list(AUTEUR_NORM_STRINGS))
    _resolved_term_ids = frozenset(name_to_id.values())
    return _resolved_term_ids


def _reset_cache_for_tests() -> None:
    """Test hook — clear the module-level cache. Not used in production."""
    global _resolved_term_ids
    _resolved_term_ids = None
