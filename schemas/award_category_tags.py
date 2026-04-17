# Three-level award category tag taxonomy.
#
# Backstory: movie_awards.category stores raw IMDB category strings (~766
# distinct). The Stage-3 award LLM endpoint cannot reliably emit ceremony-
# specific surface forms ("Best Actor in a Leading Role" vs. "Best Performance
# by an Actor in a Motion Picture - Drama"). This module collapses those raw
# strings into 62 leaf concepts, then layers two coarser tiers on top so the
# LLM can pick at whatever specificity the query implies:
#
#   level 0 (LEAF, ids 1..99)  — what consolidate() in
#       consolidate_award_categories.py returns. Every distinct concept
#       (e.g. "lead-actor", "best-picture-drama", "worst-screenplay").
#
#   level 1 (MID,  ids 100..199) — meaningful rollups across leaves
#       (e.g. "lead-acting" covers lead-actor + lead-actress;
#       "best-picture-any" covers best-picture + the genre splits;
#       "short" covers all four short-film leaves; etc.). Defined ONLY
#       where multiple leaves share a useful intermediate concept; leaves
#       with no useful rollup have no L1 parent.
#
#   level 2 (GROUP, ids 10000..10006) — the seven top-level buckets:
#       acting, directing, writing, picture, craft, razzie,
#       festival-or-tribute. Every leaf has exactly one group ancestor.
#
# Tag IDs are globally unique across the whole taxonomy via the 100^level
# scheme (1..99 / 100..199 / 10000+). Adding a 4th level later would fit
# at 1_000_000+. INT[] is plenty of headroom.
#
# Storage: movie_awards.category_tag_ids INT[] holds the leaf id + every
# ancestor id for that row. A row tagged with leaf "lead-actor" stores
# [1, 100, 10000] = [LEAD_ACTOR, LEAD_ACTING, ACTING].
#
# Query: WHERE category_tag_ids && ARRAY[...] using a GIN index. The LLM
# picks any tag(s) at any level; one indexed query handles every case.
#
# Per-level enum classes (CategoryTagL0/L1/L2) are NOT exposed; instead a
# single combined CategoryTag enum is used for the LLM-facing schema (each
# member carries a `level` attribute). This keeps the structured-output
# JSON schema simple — Pydantic + OpenAI/Anthropic structured output do
# not handle Union[Enum, Enum, Enum] cleanly. The level attribute and
# the LEVEL_*_TAGS constants below preserve per-level views for code
# that needs them (prompt rendering, validation, etc.).

from enum import Enum

from consolidate_award_categories import consolidate


# ---------------------------------------------------------------------------
# Combined enum. Members are listed leaf-first then mid then group, mirroring
# the order in which the LLM's prompt will introduce them (most-specific
# first). String value is the slug used in JSON; tag_id is the integer used
# in the Postgres array; level is 0/1/2 for runtime introspection.
# ---------------------------------------------------------------------------
class CategoryTag(str, Enum):
    tag_id: int
    level: int

    def __new__(cls, slug: str, tag_id: int, level: int) -> "CategoryTag":
        obj = str.__new__(cls, slug)
        obj._value_ = slug
        obj.tag_id = tag_id
        obj.level = level
        return obj

    # =====================================================================
    # LEVEL 0 — LEAVES (ids 1..99). One per consolidate() concept slug.
    # =====================================================================

    # Acting
    LEAD_ACTOR               = ("lead-actor", 1, 0)
    LEAD_ACTRESS             = ("lead-actress", 2, 0)
    SUPPORTING_ACTOR         = ("supporting-actor", 3, 0)
    SUPPORTING_ACTRESS       = ("supporting-actress", 4, 0)
    ENSEMBLE_CAST            = ("ensemble-cast", 5, 0)
    YOUNG_PERFORMER          = ("young-performer", 6, 0)
    BREAKTHROUGH_PERFORMANCE = ("breakthrough-performance", 7, 0)
    STUNT_PERFORMANCE        = ("stunt-performance", 8, 0)

    # Directing
    DIRECTOR                 = ("director", 9, 0)
    DEBUT_DIRECTOR           = ("debut-director", 10, 0)
    ASSISTANT_DIRECTOR       = ("assistant-director", 11, 0)

    # Writing
    SCREENPLAY               = ("screenplay", 12, 0)
    ORIGINAL_SCREENPLAY      = ("original-screenplay", 13, 0)
    ADAPTED_SCREENPLAY       = ("adapted-screenplay", 14, 0)
    DEBUT_SCREENPLAY         = ("debut-screenplay", 15, 0)

    # Picture
    BEST_PICTURE                 = ("best-picture", 16, 0)
    BEST_PICTURE_DRAMA           = ("best-picture-drama", 17, 0)
    BEST_PICTURE_COMEDY_MUSICAL  = ("best-picture-comedy-musical", 18, 0)
    BEST_PICTURE_ACTION          = ("best-picture-action", 19, 0)
    BEST_PICTURE_HORROR_SCIFI    = ("best-picture-horror-scifi", 20, 0)
    BEST_PICTURE_CRIME_ADVENTURE = ("best-picture-crime-adventure", 21, 0)
    FOREIGN_FILM             = ("foreign-film", 22, 0)
    ANIMATED_FEATURE         = ("animated-feature", 23, 0)
    DOCUMENTARY_FEATURE      = ("documentary-feature", 24, 0)
    SHORT_FILM               = ("short-film", 25, 0)
    ANIMATED_SHORT           = ("animated-short", 26, 0)
    DOCUMENTARY_SHORT        = ("documentary-short", 27, 0)
    LIVE_ACTION_SHORT        = ("live-action-short", 28, 0)
    TV_MOVIE                 = ("tv-movie", 29, 0)
    FAMILY_OR_KIDS_FILM      = ("family-or-kids-film", 30, 0)
    DEBUT_FEATURE            = ("debut-feature", 31, 0)

    # Craft
    CINEMATOGRAPHY           = ("cinematography", 32, 0)
    EDITING                  = ("editing", 33, 0)
    PRODUCTION_DESIGN        = ("production-design", 34, 0)
    COSTUME_DESIGN           = ("costume-design", 35, 0)
    MAKEUP_HAIR              = ("makeup-hair", 36, 0)
    SOUND                    = ("sound", 37, 0)
    SOUND_MIXING             = ("sound-mixing", 38, 0)
    SOUND_EDITING            = ("sound-editing", 39, 0)
    VISUAL_EFFECTS           = ("visual-effects", 40, 0)
    ORIGINAL_SCORE           = ("original-score", 41, 0)
    ORIGINAL_SONG            = ("original-song", 42, 0)
    CASTING                  = ("casting", 43, 0)
    CHOREOGRAPHY             = ("choreography", 44, 0)
    PRODUCTION_OTHER         = ("production-other", 45, 0)

    # Razzie (Worst-of-cinema awards — opposite-sentiment, kept distinct)
    WORST_PICTURE            = ("worst-picture", 46, 0)
    WORST_LEAD_ACTOR         = ("worst-lead-actor", 47, 0)
    WORST_LEAD_ACTRESS       = ("worst-lead-actress", 48, 0)
    WORST_SUPPORTING_ACTOR   = ("worst-supporting-actor", 49, 0)
    WORST_SUPPORTING_ACTRESS = ("worst-supporting-actress", 50, 0)
    WORST_DIRECTOR           = ("worst-director", 51, 0)
    WORST_SCREENPLAY         = ("worst-screenplay", 52, 0)
    WORST_REMAKE_OR_SEQUEL   = ("worst-remake-or-sequel", 53, 0)
    WORST_CAST_OR_COUPLE     = ("worst-cast-or-couple", 54, 0)
    WORST_MUSIC              = ("worst-music", 55, 0)
    WORST_VISUAL_EFFECTS     = ("worst-visual-effects", 56, 0)
    WORST_DEBUT_OR_NEWCOMER  = ("worst-debut-or-newcomer", 57, 0)
    WORST_OTHER              = ("worst-other", 58, 0)

    # Festival / jury / tribute
    FESTIVAL_SECTION         = ("festival-section", 59, 0)
    JURY_OR_SPECIAL_PRIZE    = ("jury-or-special-prize", 60, 0)
    BREAKTHROUGH_OTHER       = ("breakthrough-other", 61, 0)
    OTHER_OR_TRIBUTE         = ("other-or-tribute", 62, 0)

    # =====================================================================
    # LEVEL 1 — MID ROLLUPS (ids 100..199). Defined only where multiple
    # leaves share a meaningful intermediate concept. Leaves with no
    # rollup have no L1 parent (their tag list is just [leaf, group]).
    # =====================================================================

    # Acting rollups
    LEAD_ACTING              = ("lead-acting", 100, 1)
    SUPPORTING_ACTING        = ("supporting-acting", 101, 1)

    # Writing rollup (covers all four screenplay-related leaves; "any
    # writing award" is the canonical broad concept and there is no
    # finer-but-not-leaf distinction worth carving)
    SCREENPLAY_ANY           = ("screenplay-any", 102, 1)

    # Picture rollups
    BEST_PICTURE_ANY         = ("best-picture-any", 103, 1)
    ANIMATED                 = ("animated", 104, 1)
    DOCUMENTARY              = ("documentary", 105, 1)
    SHORT                    = ("short", 106, 1)

    # Craft rollups
    SOUND_ANY                = ("sound-any", 107, 1)
    MUSIC                    = ("music", 108, 1)
    VISUAL_CRAFT             = ("visual-craft", 109, 1)

    # Razzie rollups
    WORST_ACTING             = ("worst-acting", 110, 1)
    WORST_CRAFT              = ("worst-craft", 111, 1)

    # =====================================================================
    # LEVEL 2 — GROUPS (ids 10000..10006). Seven top-level buckets.
    # =====================================================================

    ACTING                   = ("acting", 10000, 2)
    DIRECTING                = ("directing", 10001, 2)
    WRITING                  = ("writing", 10002, 2)
    PICTURE                  = ("picture", 10003, 2)
    CRAFT                    = ("craft", 10004, 2)
    RAZZIE                   = ("razzie", 10005, 2)
    FESTIVAL_OR_TRIBUTE      = ("festival-or-tribute", 10006, 2)


# ---------------------------------------------------------------------------
# Per-level views. Built once at import time.
# ---------------------------------------------------------------------------

LEVEL_0_TAGS: tuple[CategoryTag, ...] = tuple(t for t in CategoryTag if t.level == 0)
LEVEL_1_TAGS: tuple[CategoryTag, ...] = tuple(t for t in CategoryTag if t.level == 1)
LEVEL_2_TAGS: tuple[CategoryTag, ...] = tuple(t for t in CategoryTag if t.level == 2)

# slug → CategoryTag, used by tags_for_category() to convert the string
# returned by consolidate() into the corresponding leaf member.
TAG_BY_SLUG: dict[str, CategoryTag] = {t.value: t for t in CategoryTag}


# ---------------------------------------------------------------------------
# Hierarchy: parent maps. Each leaf has a list of L1 parents (possibly
# empty) and exactly one L2 group. L1s are not modeled as having L2 parents
# in a separate map because the relationship is implicit in the rollup
# definitions below; LEAF_TO_GROUP is used for tag expansion and is the
# authoritative leaf→group map.
# ---------------------------------------------------------------------------

# Leaves that have no meaningful intermediate rollup map to an empty list.
LEAF_TO_MIDS: dict[CategoryTag, tuple[CategoryTag, ...]] = {
    # Acting
    CategoryTag.LEAD_ACTOR:               (CategoryTag.LEAD_ACTING,),
    CategoryTag.LEAD_ACTRESS:             (CategoryTag.LEAD_ACTING,),
    CategoryTag.SUPPORTING_ACTOR:         (CategoryTag.SUPPORTING_ACTING,),
    CategoryTag.SUPPORTING_ACTRESS:       (CategoryTag.SUPPORTING_ACTING,),
    CategoryTag.ENSEMBLE_CAST:            (),
    CategoryTag.YOUNG_PERFORMER:          (),
    CategoryTag.BREAKTHROUGH_PERFORMANCE: (),
    CategoryTag.STUNT_PERFORMANCE:        (),

    # Directing — no L1 mids
    CategoryTag.DIRECTOR:                 (),
    CategoryTag.DEBUT_DIRECTOR:           (),
    CategoryTag.ASSISTANT_DIRECTOR:       (),

    # Writing — single rollup over all four
    CategoryTag.SCREENPLAY:               (CategoryTag.SCREENPLAY_ANY,),
    CategoryTag.ORIGINAL_SCREENPLAY:      (CategoryTag.SCREENPLAY_ANY,),
    CategoryTag.ADAPTED_SCREENPLAY:       (CategoryTag.SCREENPLAY_ANY,),
    CategoryTag.DEBUT_SCREENPLAY:         (CategoryTag.SCREENPLAY_ANY,),

    # Picture
    CategoryTag.BEST_PICTURE:                 (CategoryTag.BEST_PICTURE_ANY,),
    CategoryTag.BEST_PICTURE_DRAMA:           (CategoryTag.BEST_PICTURE_ANY,),
    CategoryTag.BEST_PICTURE_COMEDY_MUSICAL:  (CategoryTag.BEST_PICTURE_ANY,),
    CategoryTag.BEST_PICTURE_ACTION:          (CategoryTag.BEST_PICTURE_ANY,),
    CategoryTag.BEST_PICTURE_HORROR_SCIFI:    (CategoryTag.BEST_PICTURE_ANY,),
    CategoryTag.BEST_PICTURE_CRIME_ADVENTURE: (CategoryTag.BEST_PICTURE_ANY,),
    CategoryTag.FOREIGN_FILM:             (),
    CategoryTag.ANIMATED_FEATURE:         (CategoryTag.ANIMATED,),
    CategoryTag.DOCUMENTARY_FEATURE:      (CategoryTag.DOCUMENTARY,),
    CategoryTag.SHORT_FILM:               (CategoryTag.SHORT,),
    # animated-short and documentary-short multi-parent: they're shorts
    # AND they're animated/documentary respectively. Querying by either
    # mid finds them.
    CategoryTag.ANIMATED_SHORT:           (CategoryTag.ANIMATED, CategoryTag.SHORT),
    CategoryTag.DOCUMENTARY_SHORT:        (CategoryTag.DOCUMENTARY, CategoryTag.SHORT),
    CategoryTag.LIVE_ACTION_SHORT:        (CategoryTag.SHORT,),
    CategoryTag.TV_MOVIE:                 (),
    CategoryTag.FAMILY_OR_KIDS_FILM:      (),
    CategoryTag.DEBUT_FEATURE:            (),

    # Craft
    CategoryTag.CINEMATOGRAPHY:           (CategoryTag.VISUAL_CRAFT,),
    CategoryTag.EDITING:                  (CategoryTag.VISUAL_CRAFT,),
    CategoryTag.PRODUCTION_DESIGN:        (CategoryTag.VISUAL_CRAFT,),
    CategoryTag.COSTUME_DESIGN:           (CategoryTag.VISUAL_CRAFT,),
    CategoryTag.MAKEUP_HAIR:              (CategoryTag.VISUAL_CRAFT,),
    CategoryTag.VISUAL_EFFECTS:           (CategoryTag.VISUAL_CRAFT,),
    CategoryTag.SOUND:                    (CategoryTag.SOUND_ANY,),
    CategoryTag.SOUND_MIXING:             (CategoryTag.SOUND_ANY,),
    CategoryTag.SOUND_EDITING:            (CategoryTag.SOUND_ANY,),
    CategoryTag.ORIGINAL_SCORE:           (CategoryTag.MUSIC,),
    CategoryTag.ORIGINAL_SONG:            (CategoryTag.MUSIC,),
    CategoryTag.CASTING:                  (),
    CategoryTag.CHOREOGRAPHY:             (),
    CategoryTag.PRODUCTION_OTHER:         (),

    # Razzie
    CategoryTag.WORST_PICTURE:               (),
    CategoryTag.WORST_LEAD_ACTOR:            (CategoryTag.WORST_ACTING,),
    CategoryTag.WORST_LEAD_ACTRESS:          (CategoryTag.WORST_ACTING,),
    CategoryTag.WORST_SUPPORTING_ACTOR:      (CategoryTag.WORST_ACTING,),
    CategoryTag.WORST_SUPPORTING_ACTRESS:    (CategoryTag.WORST_ACTING,),
    CategoryTag.WORST_DIRECTOR:              (),
    CategoryTag.WORST_SCREENPLAY:            (),
    CategoryTag.WORST_REMAKE_OR_SEQUEL:      (),
    CategoryTag.WORST_CAST_OR_COUPLE:        (CategoryTag.WORST_ACTING,),
    CategoryTag.WORST_MUSIC:                 (CategoryTag.WORST_CRAFT,),
    CategoryTag.WORST_VISUAL_EFFECTS:        (CategoryTag.WORST_CRAFT,),
    CategoryTag.WORST_DEBUT_OR_NEWCOMER:     (CategoryTag.WORST_ACTING,),
    CategoryTag.WORST_OTHER:                 (),

    # Festival / jury / tribute
    CategoryTag.FESTIVAL_SECTION:         (),
    CategoryTag.JURY_OR_SPECIAL_PRIZE:    (),
    CategoryTag.BREAKTHROUGH_OTHER:       (),
    CategoryTag.OTHER_OR_TRIBUTE:         (),
}

# Each L0 leaf belongs to exactly one L2 group. Authoritative source for
# tag expansion regardless of whether L1 mids exist for that leaf.
LEAF_TO_GROUP: dict[CategoryTag, CategoryTag] = {
    # Acting
    CategoryTag.LEAD_ACTOR:               CategoryTag.ACTING,
    CategoryTag.LEAD_ACTRESS:             CategoryTag.ACTING,
    CategoryTag.SUPPORTING_ACTOR:         CategoryTag.ACTING,
    CategoryTag.SUPPORTING_ACTRESS:       CategoryTag.ACTING,
    CategoryTag.ENSEMBLE_CAST:            CategoryTag.ACTING,
    CategoryTag.YOUNG_PERFORMER:          CategoryTag.ACTING,
    CategoryTag.BREAKTHROUGH_PERFORMANCE: CategoryTag.ACTING,
    CategoryTag.STUNT_PERFORMANCE:        CategoryTag.ACTING,

    # Directing
    CategoryTag.DIRECTOR:                 CategoryTag.DIRECTING,
    CategoryTag.DEBUT_DIRECTOR:           CategoryTag.DIRECTING,
    CategoryTag.ASSISTANT_DIRECTOR:       CategoryTag.DIRECTING,

    # Writing
    CategoryTag.SCREENPLAY:               CategoryTag.WRITING,
    CategoryTag.ORIGINAL_SCREENPLAY:      CategoryTag.WRITING,
    CategoryTag.ADAPTED_SCREENPLAY:       CategoryTag.WRITING,
    CategoryTag.DEBUT_SCREENPLAY:         CategoryTag.WRITING,

    # Picture
    CategoryTag.BEST_PICTURE:                 CategoryTag.PICTURE,
    CategoryTag.BEST_PICTURE_DRAMA:           CategoryTag.PICTURE,
    CategoryTag.BEST_PICTURE_COMEDY_MUSICAL:  CategoryTag.PICTURE,
    CategoryTag.BEST_PICTURE_ACTION:          CategoryTag.PICTURE,
    CategoryTag.BEST_PICTURE_HORROR_SCIFI:    CategoryTag.PICTURE,
    CategoryTag.BEST_PICTURE_CRIME_ADVENTURE: CategoryTag.PICTURE,
    CategoryTag.FOREIGN_FILM:             CategoryTag.PICTURE,
    CategoryTag.ANIMATED_FEATURE:         CategoryTag.PICTURE,
    CategoryTag.DOCUMENTARY_FEATURE:      CategoryTag.PICTURE,
    CategoryTag.SHORT_FILM:               CategoryTag.PICTURE,
    CategoryTag.ANIMATED_SHORT:           CategoryTag.PICTURE,
    CategoryTag.DOCUMENTARY_SHORT:        CategoryTag.PICTURE,
    CategoryTag.LIVE_ACTION_SHORT:        CategoryTag.PICTURE,
    CategoryTag.TV_MOVIE:                 CategoryTag.PICTURE,
    CategoryTag.FAMILY_OR_KIDS_FILM:      CategoryTag.PICTURE,
    CategoryTag.DEBUT_FEATURE:            CategoryTag.PICTURE,

    # Craft
    CategoryTag.CINEMATOGRAPHY:           CategoryTag.CRAFT,
    CategoryTag.EDITING:                  CategoryTag.CRAFT,
    CategoryTag.PRODUCTION_DESIGN:        CategoryTag.CRAFT,
    CategoryTag.COSTUME_DESIGN:           CategoryTag.CRAFT,
    CategoryTag.MAKEUP_HAIR:              CategoryTag.CRAFT,
    CategoryTag.SOUND:                    CategoryTag.CRAFT,
    CategoryTag.SOUND_MIXING:             CategoryTag.CRAFT,
    CategoryTag.SOUND_EDITING:            CategoryTag.CRAFT,
    CategoryTag.VISUAL_EFFECTS:           CategoryTag.CRAFT,
    CategoryTag.ORIGINAL_SCORE:           CategoryTag.CRAFT,
    CategoryTag.ORIGINAL_SONG:            CategoryTag.CRAFT,
    CategoryTag.CASTING:                  CategoryTag.CRAFT,
    CategoryTag.CHOREOGRAPHY:             CategoryTag.CRAFT,
    CategoryTag.PRODUCTION_OTHER:         CategoryTag.CRAFT,

    # Razzie
    CategoryTag.WORST_PICTURE:               CategoryTag.RAZZIE,
    CategoryTag.WORST_LEAD_ACTOR:            CategoryTag.RAZZIE,
    CategoryTag.WORST_LEAD_ACTRESS:          CategoryTag.RAZZIE,
    CategoryTag.WORST_SUPPORTING_ACTOR:      CategoryTag.RAZZIE,
    CategoryTag.WORST_SUPPORTING_ACTRESS:    CategoryTag.RAZZIE,
    CategoryTag.WORST_DIRECTOR:              CategoryTag.RAZZIE,
    CategoryTag.WORST_SCREENPLAY:            CategoryTag.RAZZIE,
    CategoryTag.WORST_REMAKE_OR_SEQUEL:      CategoryTag.RAZZIE,
    CategoryTag.WORST_CAST_OR_COUPLE:        CategoryTag.RAZZIE,
    CategoryTag.WORST_MUSIC:                 CategoryTag.RAZZIE,
    CategoryTag.WORST_VISUAL_EFFECTS:        CategoryTag.RAZZIE,
    CategoryTag.WORST_DEBUT_OR_NEWCOMER:     CategoryTag.RAZZIE,
    CategoryTag.WORST_OTHER:                 CategoryTag.RAZZIE,

    # Festival / jury / tribute
    CategoryTag.FESTIVAL_SECTION:         CategoryTag.FESTIVAL_OR_TRIBUTE,
    CategoryTag.JURY_OR_SPECIAL_PRIZE:    CategoryTag.FESTIVAL_OR_TRIBUTE,
    CategoryTag.BREAKTHROUGH_OTHER:       CategoryTag.FESTIVAL_OR_TRIBUTE,
    CategoryTag.OTHER_OR_TRIBUTE:         CategoryTag.FESTIVAL_OR_TRIBUTE,
}


# ---------------------------------------------------------------------------
# Tag expansion: leaf → all ancestor ids the row should be tagged with.
# Cached at import time for O(1) lookup during ingestion.
# ---------------------------------------------------------------------------

def _expand_leaf(leaf: CategoryTag) -> tuple[int, ...]:
    """Compute the sorted unique tag-id list for a leaf: leaf + mids + group."""
    if leaf.level != 0:
        raise ValueError(f"_expand_leaf expects a level-0 tag, got {leaf!r} (level {leaf.level})")
    ids: set[int] = {leaf.tag_id, LEAF_TO_GROUP[leaf].tag_id}
    ids.update(mid.tag_id for mid in LEAF_TO_MIDS[leaf])
    return tuple(sorted(ids))


_LEAF_EXPANSION: dict[CategoryTag, tuple[int, ...]] = {
    leaf: _expand_leaf(leaf) for leaf in LEVEL_0_TAGS
}


# ---------------------------------------------------------------------------
# Razzie tag set: every tag whose presence in a query's category_tags
# unambiguously signals Razzie intent. Used by the Stage-3 award executor
# to override the default Razzie ceremony exclusion when the user expressed
# Razzie intent via the tag axis instead of the ceremonies axis.
#
# Members:
#   - the RAZZIE group (10005)
#   - every leaf whose group is RAZZIE (worst-* leaves, ids 46..58)
#   - every mid that rolls up razzie leaves (worst-acting, worst-craft)
# ---------------------------------------------------------------------------
RAZZIE_TAG_IDS: frozenset[int] = frozenset(
    {CategoryTag.RAZZIE.tag_id}
    | {leaf.tag_id for leaf in LEVEL_0_TAGS if LEAF_TO_GROUP[leaf] is CategoryTag.RAZZIE}
    | {CategoryTag.WORST_ACTING.tag_id, CategoryTag.WORST_CRAFT.tag_id}
)


# ---------------------------------------------------------------------------
# Public API used by ingestion and the search execution layer.
# ---------------------------------------------------------------------------

def tags_for_leaf(leaf: CategoryTag) -> list[int]:
    """Return all tag ids (leaf + ancestors) for a level-0 tag.

    Used by ingestion to populate movie_awards.category_tag_ids.
    """
    return list(_LEAF_EXPANSION[leaf])


# ---------------------------------------------------------------------------
# Prompt rendering. Produces the human-readable taxonomy block embedded in
# the Stage-3 award LLM prompt, derived directly from the enum + hierarchy
# so the prompt can never drift from the schema.
# ---------------------------------------------------------------------------

# Short human-readable description for each tag, used in the prompt to
# disambiguate slugs. Kept terse — one short clause each.
_TAG_DESCRIPTIONS: dict[CategoryTag, str] = {
    # Acting leaves
    CategoryTag.LEAD_ACTOR:               "leading male role (Best Actor and equivalents across ceremonies)",
    CategoryTag.LEAD_ACTRESS:             "leading female role (Best Actress and equivalents)",
    CategoryTag.SUPPORTING_ACTOR:         "supporting male role",
    CategoryTag.SUPPORTING_ACTRESS:       "supporting female role",
    CategoryTag.ENSEMBLE_CAST:            "whole-cast / ensemble acting awards (e.g. SAG ensemble)",
    CategoryTag.YOUNG_PERFORMER:          "child / young actor or actress",
    CategoryTag.BREAKTHROUGH_PERFORMANCE: "first-time or breakthrough acting recognition",
    CategoryTag.STUNT_PERFORMANCE:        "stunt acting / stunt ensemble",
    # Acting mids
    CategoryTag.LEAD_ACTING:              "any leading-role acting award (gender-neutral rollup of lead-actor + lead-actress)",
    CategoryTag.SUPPORTING_ACTING:        "any supporting-role acting award (gender-neutral rollup)",

    # Directing leaves (no L1 mids)
    CategoryTag.DIRECTOR:                 "directing the film (Best Director and equivalents)",
    CategoryTag.DEBUT_DIRECTOR:           "first-feature / debut director",
    CategoryTag.ASSISTANT_DIRECTOR:       "assistant or second-unit director",

    # Writing leaves
    CategoryTag.SCREENPLAY:               "screenplay award when original-vs-adapted is unspecified",
    CategoryTag.ORIGINAL_SCREENPLAY:      "original screenplay (not adapted from prior material)",
    CategoryTag.ADAPTED_SCREENPLAY:       "adapted screenplay (from a book, play, prior film, etc.)",
    CategoryTag.DEBUT_SCREENPLAY:         "first / debut screenplay",
    # Writing mid
    CategoryTag.SCREENPLAY_ANY:           "any screenplay/writing award (rollup over the four screenplay leaves)",

    # Picture leaves
    CategoryTag.BEST_PICTURE:                 "best picture / film / motion picture (no genre split)",
    CategoryTag.BEST_PICTURE_DRAMA:           "best picture in the drama category (Globes/CCA-style split)",
    CategoryTag.BEST_PICTURE_COMEDY_MUSICAL:  "best picture in the comedy or musical category",
    CategoryTag.BEST_PICTURE_ACTION:          "best action picture",
    CategoryTag.BEST_PICTURE_HORROR_SCIFI:    "best horror or sci-fi picture",
    CategoryTag.BEST_PICTURE_CRIME_ADVENTURE: "best crime or adventure picture",
    CategoryTag.FOREIGN_FILM:             "foreign-language / international feature film",
    CategoryTag.ANIMATED_FEATURE:         "animated feature film",
    CategoryTag.DOCUMENTARY_FEATURE:      "documentary feature film",
    CategoryTag.SHORT_FILM:               "live-action or unspecified short film",
    CategoryTag.ANIMATED_SHORT:           "animated short film",
    CategoryTag.DOCUMENTARY_SHORT:        "documentary short film",
    CategoryTag.LIVE_ACTION_SHORT:        "live-action short film",
    CategoryTag.TV_MOVIE:                 "television movie / miniseries / limited series at the picture level",
    CategoryTag.FAMILY_OR_KIDS_FILM:      "family or children's film",
    CategoryTag.DEBUT_FEATURE:            "filmmaker's first / debut feature",
    # Picture mids
    CategoryTag.BEST_PICTURE_ANY:         "any best-picture award (rollup of best-picture + the five genre-split picture leaves)",
    CategoryTag.ANIMATED:                 "any animated film award (feature or short)",
    CategoryTag.DOCUMENTARY:              "any documentary film award (feature or short)",
    CategoryTag.SHORT:                    "any short-film award (live-action, animated, or documentary)",

    # Craft leaves
    CategoryTag.CINEMATOGRAPHY:           "cinematography / camera / lighting / photography",
    CategoryTag.EDITING:                  "film editing",
    CategoryTag.PRODUCTION_DESIGN:        "production design / art direction / set decoration",
    CategoryTag.COSTUME_DESIGN:           "costume design",
    CategoryTag.MAKEUP_HAIR:              "makeup and hairstyling",
    CategoryTag.SOUND:                    "sound (general / unspecified)",
    CategoryTag.SOUND_MIXING:             "sound mixing specifically",
    CategoryTag.SOUND_EDITING:            "sound editing / sound effects editing specifically",
    CategoryTag.VISUAL_EFFECTS:           "visual effects / special effects",
    CategoryTag.ORIGINAL_SCORE:           "original musical score / composing",
    CategoryTag.ORIGINAL_SONG:            "original song",
    CategoryTag.CASTING:                  "casting",
    CategoryTag.CHOREOGRAPHY:             "dance / choreography",
    CategoryTag.PRODUCTION_OTHER:         "other production-crew roles (production manager, prop master, etc.)",
    # Craft mids
    CategoryTag.SOUND_ANY:                "any sound award (general sound + sound-mixing + sound-editing)",
    CategoryTag.MUSIC:                    "any music award (original score + original song)",
    CategoryTag.VISUAL_CRAFT:             "any visual / craft award (cinematography, editing, production design, costume, makeup, VFX)",

    # Razzie leaves
    CategoryTag.WORST_PICTURE:               "Razzie worst picture",
    CategoryTag.WORST_LEAD_ACTOR:            "Razzie worst lead actor",
    CategoryTag.WORST_LEAD_ACTRESS:          "Razzie worst lead actress",
    CategoryTag.WORST_SUPPORTING_ACTOR:      "Razzie worst supporting actor",
    CategoryTag.WORST_SUPPORTING_ACTRESS:    "Razzie worst supporting actress",
    CategoryTag.WORST_DIRECTOR:              "Razzie worst director",
    CategoryTag.WORST_SCREENPLAY:            "Razzie worst screenplay",
    CategoryTag.WORST_REMAKE_OR_SEQUEL:      "Razzie worst remake / sequel / rip-off",
    CategoryTag.WORST_CAST_OR_COUPLE:        "Razzie worst screen couple / combo / ensemble",
    CategoryTag.WORST_MUSIC:                 "Razzie worst music (song / score)",
    CategoryTag.WORST_VISUAL_EFFECTS:        "Razzie worst visual effects",
    CategoryTag.WORST_DEBUT_OR_NEWCOMER:     "Razzie worst new star / debut",
    CategoryTag.WORST_OTHER:                 "Razzie miscellaneous worst categories",
    # Razzie mids
    CategoryTag.WORST_ACTING:                "any Razzie worst-acting award (lead, supporting, cast, debut)",
    CategoryTag.WORST_CRAFT:                 "any Razzie worst-craft award (music, visual effects)",

    # Festival / jury / tribute leaves
    CategoryTag.FESTIVAL_SECTION:         "festival programming section (Directors' Fortnight, Un Certain Regard, World Cinema, Panorama, Forum, etc. — programming labels, not competitive wins)",
    CategoryTag.JURY_OR_SPECIAL_PRIZE:    "festival jury prize / grand prize / special mention (a competitive win that isn't a conventional category)",
    CategoryTag.BREAKTHROUGH_OTHER:       "breakthrough recognition for non-acting roles (director, filmmaker, artist)",
    CategoryTag.OTHER_OR_TRIBUTE:         "honorary / tribute / artistic-contribution recognitions",

    # Top-level groups
    CategoryTag.ACTING:              "any acting award at all (covers every acting leaf and mid)",
    CategoryTag.DIRECTING:           "any directing award (covers director, debut-director, assistant-director)",
    CategoryTag.WRITING:             "any writing/screenplay award",
    CategoryTag.PICTURE:             "any picture-format award (best-picture, foreign, animated, documentary, short, tv-movie, family, debut feature)",
    CategoryTag.CRAFT:               "any craft / technical / music award",
    CategoryTag.RAZZIE:              "any Razzie award (worst-of-cinema)",
    CategoryTag.FESTIVAL_OR_TRIBUTE: "any festival section / jury prize / honorary tribute recognition",
}


# Fail loudly at import time if a tag is added without a description.
# Without this, a missing entry would only surface the first time the
# prompt is assembled (KeyError deep inside render_taxonomy_for_prompt).
_missing_descriptions = [t for t in CategoryTag if t not in _TAG_DESCRIPTIONS]
assert not _missing_descriptions, (
    f"_TAG_DESCRIPTIONS is missing entries for: {_missing_descriptions}"
)


def render_taxonomy_for_prompt() -> str:
    """Render the full 3-level tag taxonomy as a printable text block.

    The block is grouped by L2 group → L1 mids → L0 leaves under the group.
    Each line lists the slug and a short description so the LLM can pick at
    any specificity. Used by the Stage-3 award prompt; defined here so the
    prompt and the enum can't drift apart.
    """
    lines: list[str] = []
    # Group tags by their L2 ancestor for layout.
    leaves_by_group: dict[CategoryTag, list[CategoryTag]] = {g: [] for g in LEVEL_2_TAGS}
    for leaf in LEVEL_0_TAGS:
        leaves_by_group[LEAF_TO_GROUP[leaf]].append(leaf)

    # Mids belong to a group via any of their child leaves; any leaf works
    # because all children of a mid share the same group.
    mids_by_group: dict[CategoryTag, list[CategoryTag]] = {g: [] for g in LEVEL_2_TAGS}
    for mid in LEVEL_1_TAGS:
        # find any leaf whose mids include this mid; use its group
        for leaf, mids in LEAF_TO_MIDS.items():
            if mid in mids:
                group = LEAF_TO_GROUP[leaf]
                if mid not in mids_by_group[group]:
                    mids_by_group[group].append(mid)
                break

    for group in LEVEL_2_TAGS:
        lines.append(f"GROUP: {group.value}")
        lines.append(f"  {group.value}  — {_TAG_DESCRIPTIONS[group]}")
        if mids_by_group[group]:
            lines.append("  rollups:")
            for mid in mids_by_group[group]:
                lines.append(f"    {mid.value}  — {_TAG_DESCRIPTIONS[mid]}")
        lines.append("  leaves:")
        for leaf in leaves_by_group[group]:
            mid_slugs = [m.value for m in LEAF_TO_MIDS[leaf]]
            mid_suffix = f"  [under: {', '.join(mid_slugs)}]" if mid_slugs else ""
            lines.append(f"    {leaf.value}  — {_TAG_DESCRIPTIONS[leaf]}{mid_suffix}")
        lines.append("")
    return "\n".join(lines).rstrip()


def tags_for_category(raw_category: str | None) -> list[int]:
    """Return all tag ids that apply to a raw movie_awards.category value.

    Pipeline: raw string → consolidate() → leaf slug → CategoryTag leaf
    member → expanded ancestor id list. Returns an empty list when the
    category is empty/None or no rule matches (defensive — empirically
    every non-empty category in the current DB maps).
    """
    if not raw_category:
        return []
    slug = consolidate(raw_category)
    if slug is None:
        return []
    leaf = TAG_BY_SLUG.get(slug)
    if leaf is None or leaf.level != 0:
        # consolidate() returned a slug we don't know about — should never
        # happen if this module and consolidate_award_categories.py stay
        # in sync, but fail soft so an unknown slug doesn't break ingest.
        return []
    return list(_LEAF_EXPANSION[leaf])
