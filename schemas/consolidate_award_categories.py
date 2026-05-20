"""
Consolidate movie_awards.category strings into a minimal canonical taxonomy.

Raw movie_awards.category values are highly variable (766 distinct strings
across ~36k rows). An LLM cannot reliably generate the exact-match string for
a given award concept from a natural-language query. This module collapses
those variants into 62 canonical concept slugs (e.g. "Best Actor",
"Best Performance by a Male Actor in a Motion Picture - Drama",
"Best Male Lead", "Outstanding Performance by a Male Actor in a Leading Role"
all map to `lead-actor`).

How it works:
  1. `norm()` lowercases, normalizes punctuation (treats -, /, , as spaces),
     and collapses whitespace.
  2. `RULES` is an ordered list of (regex, concept) pairs. The first rule
     whose regex matches the normalized text wins. Order is critical:
       - specific qualifiers (supporting, lead, debut) must be tested before
         generic role words (actor, director)
       - craft rules (editing, score, sound) must be tested before the
         generic best-picture/film catchall, otherwise "Best Film Editing"
         would be bucketed as best-picture.
  3. `consolidate(category)` returns the canonical concept slug, or None
     if no rule matched (empirically, 100% of non-empty rows are mapped).

Design choices:
  - Lead vs supporting, actor vs actress kept distinct (4 separate concepts)
    because every major ceremony awards them separately.
  - Polarity baked into the slug: best-* vs worst-* are separate concepts.
    Razzies and Oscars are opposite intent and should not merge.
  - Best Picture split by genre (drama, comedy-musical, action, horror-scifi,
    crime-adventure) because Golden Globes / Critics' Choice split them.
  - Format splits: foreign-film, animated-feature/short, documentary-
    feature/short, live-action-short, short-film, tv-movie, family-or-kids,
    debut-feature. These are real distinctions in most award ceremonies.
  - Sound split into three (sound, sound-mixing, sound-editing) because the
    Academy awards them separately.
  - Score and song kept separate (also separate Oscars).
  - Festival sections (Directors' Fortnight, Un Certain Regard, World Cinema,
    Panorama, Forum, NEXT, Spotlight) are programming labels, not competitive
    wins — bucketed as `festival-section` rather than picture-level concepts.
  - `screenplay` is the catchall for writing awards that don't specify
    original vs adapted ("Best Screenplay" unqualified is genuinely ambiguous).

Known judgment calls:
  - TV-movie acting awards (miniseries, limited-series roles) collapse into
    `lead-actor` / `supporting-actress` / etc. rather than a TV-specific
    acting bucket. The `tv-movie` concept only catches picture-level TV-movie
    awards. Rationale: this is a movie finder — the TV-movie distinction is
    meaningful at the picture level, but acting is acting.

Usage:
  from consolidate_award_categories import consolidate

  concept = consolidate("Best Performance by an Actor in a Leading Role")
  # -> "lead-actor"

Running as a script prints the full taxonomy with per-concept row counts and
example variants; useful for auditing and regression-checking.
"""
import os
import re
from collections import defaultdict

import psycopg
from dotenv import load_dotenv

load_dotenv()


def fetch_categories():
    """Return [(category_text, row_count)] for all non-empty categories."""
    with psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT category, COUNT(*) c "
                "FROM movie_awards "
                "WHERE category IS NOT NULL AND category <> '' "
                "GROUP BY category ORDER BY c DESC"
            )
            return cur.fetchall()


def norm(s: str) -> str:
    """Normalize a raw category string for rule matching.

    - Lowercases, strips.
    - Converts smart quotes to ASCII apostrophe.
    - Treats `-`, `/`, `,` as word separators (folds to spaces).
    - Collapses runs of whitespace.
    """
    s = s.lower().strip()
    s = re.sub(r"[\u2018\u2019\u201c\u201d`]", "'", s)
    s = re.sub(r"[-/,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Composable token classes. Text is already normalized (lower-case, no
# commas/dashes) by the time a rule runs.
WORST = r"(worst|most flatulent|so rotten|reckless disregard)"
ACTOR_M = r"(actor|\bmale\b|\bman\b|\bmen\b)"
ACTOR_F = r"(actress|\bfemale\b|\bwoman\b|\bwomen\b)"
ACTING = r"(actor|actress|performer|performance|cast\b|acting)"
SUPP = r"(support)"
TVMOVIE = (r"(television|\btv\b|miniseries|mini series|limited series|"
           r"made for tv|made for television|tv movie|television movie)")

# Ordered rule list. First match wins.
RULES: list[tuple[re.Pattern, str]] = []


def _add(pattern: str, concept: str) -> None:
    RULES.append((re.compile(pattern), concept))


# ============================================================================
# 1. WORST (Razzies) — polarity is baked into the concept slug.
#    Checked first so that specific negative awards don't accidentally match
#    the positive "lead-actor" / "best-picture" rules below.
# ============================================================================
_add(rf"{WORST}.*screen (couple|combo|ensemble)", "worst-cast-or-couple")
_add(rf"{WORST}.*ensemble", "worst-cast-or-couple")
_add(rf"{WORST}.*supporting.*{ACTOR_F}", "worst-supporting-actress")
_add(rf"{WORST}.*supporting.*{ACTOR_M}", "worst-supporting-actor")
_add(rf"{WORST}.*{ACTOR_F}", "worst-lead-actress")
_add(rf"{WORST}.*{ACTOR_M}", "worst-lead-actor")
_add(rf"{WORST}.*director", "worst-director")
_add(rf"{WORST}.*screenplay|written.*film grossing", "worst-screenplay")
_add(rf"{WORST}.*(remake|sequel|prequel|rip off)", "worst-remake-or-sequel")
_add(rf"{WORST}.*visual effects", "worst-visual-effects")
_add(rf"{WORST}.*(song|score|music)", "worst-music")
_add(rf"{WORST}.*new star", "worst-debut-or-newcomer")
_add(rf"{WORST}.*(picture|film|movie|drama|comedy|musical|entertainment|horror|3 ?d)",
     "worst-picture")
_add(rf"{WORST}", "worst-other")


# ============================================================================
# 2. ACTING (positive polarity) — most-specific first.
#    Order within the section matters: stunt > ensemble > young > breakthrough >
#    supporting > lead, otherwise a generic "actor" rule would swallow
#    more-specific ones.
# ============================================================================
# Stunt performance — checked before ensemble because stunt awards often say
# "Stunt Ensemble".
_add(r"stunt", "stunt-performance")

# Cast / ensemble (gender-neutral, no lead/support distinction).
_add(r"\b(cast|ensemble|acting team)\b", "ensemble-cast")

# Young / child performer — age modifier overrides the generic actor rule.
_add(r"(young|child|children).*(actor|actress|performer|performance|lead)",
     "young-performer")
_add(r"new young (actor|actress)", "young-performer")

# Debut / breakthrough acting performances (a person's first-time recognition).
_add(rf"(debut|breakthrough|break through|newcomer|new star).*({ACTING})",
     "breakthrough-performance")
_add(rf"({ACTING}).*(debut|breakthrough|newcomer|emerging|revelation)",
     "breakthrough-performance")
_add(r"(most )?promising newcomer", "breakthrough-performance")
_add(r"new star of the year", "breakthrough-performance")
_add(r"acting debut", "breakthrough-performance")
_add(r"most outstanding newcomer", "breakthrough-performance")

# Breakthrough recognitions that are NOT acting (director/filmmaker/artist).
_add(r"breakthrough (performer|performance|artist|talent|filmmaker|director)",
     "breakthrough-other")
_add(r"break through", "breakthrough-other")
_add(r"emerging (filmmaker|talent)", "breakthrough-other")

# Foreign / national prefixes on acting awards still resolve to lead role.
# "Best Foreign Actor" / "Best British Actress" => lead-actor / lead-actress.
_add(rf"foreign.*{ACTOR_F}|british.*{ACTOR_F}", "lead-actress")
_add(rf"foreign.*{ACTOR_M}|british.*{ACTOR_M}", "lead-actor")

# Supporting roles (gendered). Checked before lead so phrasings like
# "Best Actor in a Supporting Role" route to supporting, not lead.
_add(rf"{SUPP}.*{ACTOR_F}", "supporting-actress")
_add(rf"{SUPP}.*{ACTOR_M}", "supporting-actor")
_add(rf"{ACTOR_F}.*{SUPP}", "supporting-actress")
_add(rf"{ACTOR_M}.*{SUPP}", "supporting-actor")
# Gender-neutral supporting — falls to supporting-actor as the default.
_add(r"supporting performance|supporting performer|supporting artist",
     "supporting-actor")

# Leading roles (gendered). Prefer genre/context anchors so ambiguous phrases
# resolve correctly, then fall back to role-word matches.
_add(rf"{ACTOR_F}.*(comedy|musical|drama|action|tv movie|television movie|"
     rf"miniseries|limited series|movie made|picture made|leading|lead)",
     "lead-actress")
_add(rf"{ACTOR_M}.*(comedy|musical|drama|action|tv movie|television movie|"
     rf"miniseries|limited series|movie made|picture made|leading|lead)",
     "lead-actor")
_add(r"\bmale lead\b|leading male", "lead-actor")
_add(r"\bfemale lead\b|leading female", "lead-actress")
_add(rf"{ACTOR_F}", "lead-actress")
_add(r"\bactor\b|\bactor in\b|leading actor|lead actor|male actor|"
     r"male performance|male performer", "lead-actor")
# Gender-neutral lead performance (e.g. Spirit Awards' "Best Lead Performance").
_add(r"leading performance|lead performance|outstanding lead performance",
     "lead-actor")
# Last-resort acting bucket — catches "Acting", "Achievement in Acting", etc.
_add(r"\bperformance\b|\bperformer\b|\bacting\b", "lead-actor")


# ============================================================================
# 3. CRAFT — placed BEFORE the picture/film catchall so that e.g.
#    "Best Film Editing" routes to `editing` and
#    "Best Original Score - Motion Picture" routes to `original-score`
#    rather than being swallowed by the generic best-picture rule.
#    Within this section, specific sub-disciplines (sound-editing, sound-mixing)
#    precede their general form (sound).
# ============================================================================
# Production design / art direction — placed BEFORE the directing rule so
# "Best Art Direction" doesn't match on the `\bdirection\b` token below.
_add(r"production design|art direction|set decoration|set design|art design|"
     r"costume and production design|graphics|animation design|^best design$|"
     r"^design$", "production-design")

_add(r"costume|dressmaker", "costume-design")

_add(r"make.?up|hairstyling|hair (and|&) make|make up artist|hair design|"
     r"make up & hair", "makeup-hair")
_add(r"\bhair\b", "makeup-hair")

# Sound sub-disciplines (BEFORE generic editing so "Sound Editing" doesn't
# land in the editing bucket).
_add(r"sound mix", "sound-mixing")
_add(r"sound edit|sound effects edit|sound effects|sound supervisor|"
     r"sound effects editing", "sound-editing")

# Visual / special effects (BEFORE editing because some categories say
# "Visual Effects Editing").
_add(r"visual effects|special effects|special visual effects|"
     r"engineering effects|3 ?d film stereoscopic|stereoscopic|effects.*sound|"
     r"effects.*visual|effects.*special|special.*visual.*graphic|"
     r"visual and graphic", "visual-effects")

# Cinematography / camera / lighting / photography.
_add(r"cinematograph|photography|cameraman|camera operator|chief electrician|"
     r"head of (camera|electrical)|lighting|video lighting|television lighting|"
     r"electrical department|color\b.*film|black and white|colour|"
     r"^best color$", "cinematography")

# Generic editing (after sound-edit / visual-fx variants have been consumed).
_add(r"\bediting\b|\beditor\b|\beditors\b|editing fiction|film editor",
     "editing")

# Music: original-song must precede original-score because some categories
# explicitly mention "Song" within a "Score" phrase (e.g. "Original Song Score").
_add(r"original (song|musical|comedy) score|scoring.*song|original song score|"
     r"\bsong\b|original song", "original-song")
_add(r"score|original music|musical score|film music|composer|"
     r"original film score|substantially original|scoring|"
     r"music written for motion pictures|original dramatic score|"
     r"\bmusic\b|synchronised music|soundtrack", "original-score")

# Generic sound (after sound-editing / sound-mixing have been consumed).
_add(r"\bsound\b|sound recording|sound track", "sound")

_add(r"casting", "casting")
_add(r"choreograph|dance direction", "choreography")

# Crew / production roles that don't map to design or direction.
_add(r"prop maker|stagehand|production manager|production director",
     "production-other")


# ============================================================================
# 4. DIRECTING — checked after production-design/art-direction so that the
#    generic `\bdirection\b` / `\bdirector\b` patterns don't swallow
#    "Best Art Direction".
# ============================================================================
# Festival sections that contain "director" (Directors' Fortnight) must be
# caught before the generic director rule fires.
_add(r"directors'? fortnight|director'?s fortnight", "festival-section")

_add(r"first.*director|new director|debut.*director|director of a debut",
     "debut-director")
_add(r"assistant director|second unit director", "assistant-director")
_add(r"\bdirector\b|directing|\bdirection\b", "director")


# ============================================================================
# 5. WRITING
# ============================================================================
_add(r"first.*(screenplay|writer|written)|debut.*screenplay|new writer",
     "debut-screenplay")
_add(r"(adapted|adaptation|based on|previously produced|previously published|"
     r"other material|another medium).*(screenplay|writing|written)",
     "adapted-screenplay")
_add(r"screenplay.*(adapted|adaptation|based on|other material|previously)",
     "adapted-screenplay")
# "Cinematic Transposition" is French-style phrasing for adaptation.
_add(r"cinematic transposition|transposition cinéma", "adapted-screenplay")
_add(r"(original|directly for the screen|not previously published)."
     r"*(screenplay|writing|written|story)", "original-screenplay")
_add(r"\b(story|motion picture story)\b", "original-screenplay")
# Catchall for writing awards that don't specify original vs adapted.
_add(r"screenplay|screen play|writing|\bwriter\b|\bscript\b|"
     r"screenwriter|screenwriting|directing and screenwriting", "screenplay")


# ============================================================================
# 6. PICTURE / FILM CATEGORIES
#    Order: specialized formats (animated, documentary, short) first, then
#    foreign / tv-movie / family, then genre-split best-picture, then the
#    generic best-picture catchall.
# ============================================================================
# Animated short must come before animated-feature and before generic short.
_add(r"animated short|short.*animat|animation.*short|short animat",
     "animated-short")
_add(r"(animated|animation).*(feature|film|movie|picture|featured film)",
     "animated-feature")
_add(r"(feature|film|movie|picture).*animated", "animated-feature")
_add(r"\banimation\b", "animated-feature")

# Documentary short before documentary-feature.
_add(r"documentary.*(short|short subject|short film)|short.*documentary",
     "documentary-short")
_add(r"(documentary|non.?fiction|essay film|cultural film|factual).*"
     r"(feature|film|movie|programme|series|cinema)", "documentary-feature")
_add(r"feature.*documentary", "documentary-feature")
_add(r"\bdocumentary\b|\bdocumentaries\b|\bnon.?fiction\b|\bfactual\b|"
     r"current affairs|specialist factual|documentary award|"
     r"investigative documentary|archival storytelling|verite filmmaking",
     "documentary-feature")

# Live-action short before generic short.
_add(r"live action.*short|short.*live action", "live-action-short")
_add(r"short.*(subject|fiction|comedy|color|two reel|one reel|novelty)",
     "short-film")
_add(r"\bshort films?\b|\bshorts\b|short film|fictional short|short fictional",
     "short-film")
_add(r"\bshort\b", "short-film")

# Foreign / international film.
_add(r"(foreign|international|non.?english|english.?language foreign|"
     r"euro.?mediterranean|italian|french|latin american|euro)\b.*"
     r"(film|feature|picture|movie|programme|cinema|director|actor in|cineaste)",
     "foreign-film")
_add(r"(film|feature|picture|movie).*(foreign|international|non.?english|"
     r"in a foreign language|not in the english language)", "foreign-film")
_add(r"foreign language|international feature|world cinema|foreign film|"
     r"foreign tv programme|foreign cineaste|french film|italian film|"
     r"french cineaste|italian director|italian actor|italian.*venice|"
     r"euro.?mediterranean", "foreign-film")

# Made-for-TV movie (only catches picture-level TV-movie awards — acting
# awards for TV-movies fall through to lead/supporting buckets above).
_add(rf"(motion picture|movie|picture|film).*({TVMOVIE})", "tv-movie")
_add(rf"({TVMOVIE}).*(motion picture|movie|picture|film)", "tv-movie")
_add(rf"{TVMOVIE}", "tv-movie")

# Family / kids / youth film.
_add(r"family film|children'?s (film|feature|programme|jury|craft|& family)|"
     r"kids|family entertainment|fairy tale|adolescence|school drama|"
     r"feature film suitable for young people|young people|youth jury",
     "family-or-kids-film")

# Genre-split best-picture variants (Golden Globes / Critics' Choice style).
_add(r"(picture|film|movie|motion picture|feature)\b.*(drama|dramatic)|"
     r"(drama|dramatic).*(film|picture|movie|motion picture|feature)",
     "best-picture-drama")
_add(r"(picture|film|movie|motion picture|feature)\b.*(comedy|musical)|"
     r"(comedy|musical).*(film|picture|movie|motion picture|feature)|"
     r"\bmusical\b", "best-picture-comedy-musical")
_add(r"(picture|film|movie|motion picture|feature)\b.*action|"
     r"action.*(film|picture|movie|motion picture|feature)",
     "best-picture-action")
_add(r"(picture|film|movie|motion picture|feature)\b.*(horror|sci.?fi)|"
     r"(sci.?fi|horror).*(film|picture|movie|motion picture|feature)",
     "best-picture-horror-scifi")
_add(r"(picture|film|movie|motion picture|feature)\b.*(crime|adventure)|"
     r"(crime|adventure).*(film|picture|movie|motion picture|feature)|"
     r"crime or adventure", "best-picture-crime-adventure")

# Debut / first feature (a filmmaker's first work).
_add(r"first feature|first film|debut film|debut feature|first work|"
     r"first.*feature|new young filmmaker|debut by", "debut-feature")

# Generic best picture catchall (runs after every specialized picture rule).
_add(r"\b(picture|film|movie|motion picture|feature|color film|adventure film|"
     r"comedy film|dramatic film|entertainment film|fiction film|"
     r"single play|single drama|drama production|drama serial|"
     r"television series|drama|comedy)\b", "best-picture")


# ============================================================================
# 7. FESTIVAL SECTIONS — non-competitive section labels (Cannes, Berlin,
#    Sundance, Venice). These are programming categories, not wins.
# ============================================================================
_add(r"(directors'? fortnight|critics'? week|quinzaine|un certain regard|"
     r"panorama|forum|world cinema|generation|encounters|spotlight|"
     r"festival favorite|venice days|venice horizons|orizzonti|berlinale|"
     r"cannes ecrans juniors|parallel sections?|alternatives|\bnext\b|"
     r"competition|in competition|out of competition|forum section|"
     r"cinéma des étudiants|teddy|teddy ballot|männer|gan foundation|"
     r"international critics week|international fiction|us fiction|"
     r"u\.s\.? (dramatic|documentary|fiction)|festival prize|"
     r"audience poll|short filmmaking|debut feature|guild film prize|"
     r"international jury|verite filmmaking|amnesty international|"
     r"berlinale efm|impact for change|impact and change|specialised programme|"
     r"glashütte original|arca|interfilm|interreligious|"
     r"^dramatic$|^international$|^national$|^fiction$|^vanguard$|"
     r"^revelation$|^perspectives$|^no borders$|^neorealism$|"
     r"^international prize$|^international critics prize$|"
     r"^collateral award$|^color$|^dvd$|"
     r"huw wheldon)\b", "festival-section")

# Jury / special prizes (festival accolades that ARE wins, just not in a
# conventional category).
_add(r"grand prix|grand jury prize|jury (prize|award|grand prix|special mention)|"
     r"grand prize|special (jury|prize|mention|distinction|award|pasinetti)|"
     r"honorary|alfred bauer|graffetta|nave d'argento|jury\b",
     "jury-or-special-prize")


# ============================================================================
# 8. MISC TRIBUTES / ARTISTIC RECOGNITIONS — tail bucket for anything that
#    doesn't fit a specific craft/role but is still a competitive recognition.
# ============================================================================
_add(r"interactivity|cinema man nature|human document|"
     r"outstanding (artistic|technical|single) achievement|"
     r"creative (vision|collaboration|storytelling)|originality|innovation|"
     r"innovative|nonfiction (experimentation|storytelling)|social impact|"
     r"social justice|art of change|activist|freedom of expression|"
     r"clarity of vision|visionary|uncompromising|cultural icon|global icon|"
     r"historical icon|icon and creator|ambassador of hope|cinema for peace|"
     r"creative vision|creative storytelling|breakthrough filmmaker|auteur|"
     r"cinematic innovation|treatment of issues|cinema, man, nature|"
     r"film on the relationship man|most touching|most amusing|most convincing|"
     r"audience award|popular movie|favorite film franchise|favorite|"
     r"food scene|inanimate object|live event|comedy programme|stand up comedy|"
     r"comedy special|specialist factual|entertainment|situation comedy|"
     r"poetic humor|advertising film|social film|psychological and love|"
     r"original story|cinematic and box office achievement|technical perfection|"
     r"british film|outstanding british film|film promoting|"
     r"artistic contribution|technical contribution|distribution|spotlight",
     "other-or-tribute")


# ============================================================================
# Public API
# ============================================================================
def consolidate(category: str) -> str | None:
    """Return the canonical concept slug for a raw `movie_awards.category`.

    Returns None if no rule matches. Empirically, 100% of non-empty categories
    in the current database map to a concept.
    """
    if not category:
        return None
    text = norm(category)
    for pat, concept in RULES:
        if pat.search(text):
            return concept
    return None


# All canonical concepts, grouped for documentation and for downstream code
# that wants to enumerate / validate the taxonomy. Order within a group is
# rough importance / frequency; groups themselves follow the pipeline order
# used above.
CONCEPTS_BY_GROUP: dict[str, list[str]] = {
    "acting": [
        "lead-actor", "lead-actress",
        "supporting-actor", "supporting-actress",
        "ensemble-cast", "young-performer",
        "breakthrough-performance", "stunt-performance",
    ],
    "directing": ["director", "debut-director", "assistant-director"],
    "writing": [
        "screenplay", "original-screenplay",
        "adapted-screenplay", "debut-screenplay",
    ],
    "picture": [
        "best-picture",
        "best-picture-drama", "best-picture-comedy-musical",
        "best-picture-action", "best-picture-horror-scifi",
        "best-picture-crime-adventure",
        "foreign-film", "animated-feature", "documentary-feature",
        "short-film", "animated-short", "documentary-short",
        "live-action-short", "tv-movie", "family-or-kids-film",
        "debut-feature",
    ],
    "craft": [
        "cinematography", "editing", "production-design", "costume-design",
        "makeup-hair", "sound", "sound-mixing", "sound-editing",
        "visual-effects", "original-score", "original-song",
        "casting", "choreography", "production-other",
    ],
    "razzie": [
        "worst-picture", "worst-lead-actor", "worst-lead-actress",
        "worst-supporting-actor", "worst-supporting-actress",
        "worst-director", "worst-screenplay",
        "worst-remake-or-sequel", "worst-cast-or-couple",
        "worst-music", "worst-visual-effects",
        "worst-debut-or-newcomer", "worst-other",
    ],
    "festival_or_tribute": [
        "festival-section", "jury-or-special-prize",
        "breakthrough-other", "other-or-tribute",
    ],
}

ALL_CONCEPTS: list[str] = [c for cs in CONCEPTS_BY_GROUP.values() for c in cs]


# ============================================================================
# CLI: audit the current database and print per-concept coverage.
# ============================================================================
def _main():
    rows = fetch_categories()
    by_concept: dict[str, list[tuple[int, str]]] = defaultdict(list)
    unmapped: list[tuple[int, str]] = []
    for cat, count in rows:
        concept = consolidate(cat)
        if concept is None:
            unmapped.append((count, cat))
        else:
            by_concept[concept].append((count, cat))

    mapped_rows = sum(c for vs in by_concept.values() for c, _ in vs)
    unmapped_rows = sum(c for c, _ in unmapped)
    total_rows = mapped_rows + unmapped_rows
    print(f"Total distinct (non-empty) categories: {len(rows)}")
    print(f"Total rows mapped:    {mapped_rows} ({mapped_rows*100/total_rows:.1f}%)")
    print(f"Total rows unmapped:  {unmapped_rows}")
    print(f"Total concepts:       {len(by_concept)}")
    print()
    print("=" * 80)
    print("CONCEPTS (sorted by row coverage)")
    print("=" * 80)
    for concept in sorted(by_concept, key=lambda k: -sum(c for c, _ in by_concept[k])):
        members = by_concept[concept]
        rows_ct = sum(c for c, _ in members)
        print(f"\n[{concept}]  rows={rows_ct}  variants={len(members)}")
        for c, cat in sorted(members, reverse=True)[:8]:
            print(f"    {c:>5}  {cat}")
        if len(members) > 8:
            print(f"      ... and {len(members) - 8} more")

    if unmapped:
        print()
        print("=" * 80)
        print(f"UNMAPPED ({len(unmapped)} distinct, {unmapped_rows} rows)")
        print("=" * 80)
        for c, cat in sorted(unmapped, reverse=True):
            print(f"    {c:>5}  {cat}")


if __name__ == "__main__":
    _main()
