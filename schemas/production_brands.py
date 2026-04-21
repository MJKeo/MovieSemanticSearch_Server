# Production-brand registry for the studio resolver.
#
# Curation principle: a label belongs in a brand's roster ONLY IF a casual
# movie watcher typing "<brand> movies" would expect its films to appear.
# The earlier iteration used "catalog recall over label purity" (every
# corporate subsidiary counted). That leaked cross-brand: "Disney movies"
# returned Miramax indies like No Country for Old Men because Disney owned
# Miramax 1993-2010. The current registry applies a brand-identity test
# instead of a corporate-ownership one:
#   - KEEP: labels the parent actively brand-promotes (e.g., Pixar as
#     "Disney/Pixar", Marvel Studios, Lucasfilm/Star Wars).
#   - DROP: autonomous-identity acquisitions deliberately kept separate
#     (Miramax, Searchlight, Touchstone, Hollywood Pictures, Blue Sky,
#     New Line, HBO Films, DC under WB, Focus, DreamWorks Animation,
#     Working Title under Universal, Nickelodeon and MTV under Paramount,
#     Sony Pictures Classics). Each remains findable via its own
#     standalone brand where one exists in the registry.
#   - DROP: home-entertainment-only, distribution-only, foreign-region,
#     and pure legal-entity credits with no user-search value.
#
# Shape:
#   - `ProductionBrand` is a str-backed enum; the enum value is the brand
#     slug (e.g. "disney"). Each member also carries `brand_id` (int) and
#     `display_name` (str) attributes.
#   - `ProductionBrand.companies` is a tuple of `BrandCompany` rows. Each
#     row is an exact IMDB `production_companies` string plus the
#     (start_year, end_year) window during which that string SHOULD count
#     as this brand.
#
# Year conventions:
#   - start_year / end_year are **inclusive**. `None` means "no bound".
#   - `(None, None)` = "always applicable for this brand". Used for strings
#     that can't be pinned to a founding/acquisition date (umbrella credit
#     strings like "Sony Pictures Entertainment", bare "A24", etc.).
#   - A single surface string may appear under multiple brands with
#     different windows (e.g. `Metro-Goldwyn-Mayer (MGM)` → MGM 1924-,
#     AMAZON_MGM 2022-). The resolver emits every brand that passes the
#     year check.
#   - A single surface string may also appear TWICE within one brand if
#     that brand had non-overlapping active eras (e.g. UA's
#     `United Artists` has both 1919-1981 and 2024- rows). The resolver
#     handles this correctly — each entry independently passes or fails
#     the year check.
#
# Lookup:
#   - `memberships_for_string(s)` returns `[(ProductionBrand, start_year,
#     end_year), ...]` sorted alphabetically by brand name (ascending),
#     then by start_year (ascending, None as -inf), then by end_year. This
#     order gives callers a deterministic iteration sequence.
#   - `year_matches(start, end, release_year)` encodes the window
#     predicate. If release_year is None, only `(None, None)` rows match —
#     any window is skipped per the ingestion-time rule.
#
# Ingestion side wires this into `movie_ingestion/final_ingestion/
# brand_resolver.py`, which takes an IMDB `production_companies` list + a
# release year and returns a list of `BrandTag(brand_id, first_matching_index)`.

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class BrandCompany:
    """One surface-string → brand relationship with its time window.

    `string` is the exact IMDB `production_companies` value as it appears
    in the database (case- and whitespace-sensitive). Matching uses exact
    string comparison at this stage; any normalization layer would sit on
    top.

    `start_year` / `end_year` are inclusive and open-ended when None. See
    the module docstring for the "always applicable" (None, None)
    convention.
    """

    string: str
    start_year: int | None
    end_year: int | None


class ProductionBrand(str, Enum):
    # Attribute annotations — attached via __new__ below. Declared here so
    # type checkers and IDEs see them on every enum member.
    brand_id: int
    display_name: str
    companies: tuple[BrandCompany, ...]

    def __new__(
        cls,
        slug: str,
        brand_id: int,
        display_name: str,
        companies: tuple[BrandCompany, ...],
    ) -> "ProductionBrand":
        # Enum value is the slug (string); the other fields are attached
        # as attributes so each member is a hashable string carrying extra
        # data.
        obj = str.__new__(cls, slug)
        obj._value_ = slug
        obj.brand_id = brand_id
        obj.display_name = display_name
        obj.companies = companies
        return obj

    # =====================================================================
    # Tier 1 — 24 major brands
    # =====================================================================

    # DISNEY — The Walt Disney Studios as casual viewers understand it:
    # Walt Disney Pictures + Animation canon, plus co-branded Pixar,
    # Marvel Studios, and Lucasfilm/Star Wars. Excluded: Fox/20th Century
    # (kept distinct by Disney), Searchlight (autonomous prestige brand),
    # Touchstone and Hollywood Pictures (created specifically to release
    # non-Disney-branded films), Miramax and Dimension (Weinstein-era
    # autonomous), Blue Sky (Fox-branded). All remain findable via their
    # own standalone brands.
    DISNEY = ("disney", 1, "The Walt Disney Studios", (
        # Walt Disney Pictures (1983-)
        BrandCompany("Walt Disney Pictures", 1983, None),
        BrandCompany("Walt Disney Studios", 1983, None),
        BrandCompany("Walt Disney Studios Motion Pictures", 1983, None),
        BrandCompany("Walt Disney British Films", 1983, None),
        # Walt Disney Productions (1929-1986)
        BrandCompany("Walt Disney Productions", 1929, 1986),
        # Walt Disney Animation Studios lineage (1923-)
        BrandCompany("Walt Disney Animation Studios", 1923, None),
        BrandCompany("Walt Disney Feature Animation", 1923, None),
        BrandCompany("Walt Disney Animation Australia", 1923, None),
        BrandCompany("Walt Disney Animation Japan", 1923, None),
        BrandCompany("Walt Disney Feature Animation Florida", 1923, None),
        BrandCompany("Walt Disney Animation Canada", 1923, None),
        BrandCompany("Walt Disney Animation France S.A.", 1923, None),
        BrandCompany("Walt Disney Feature Animation Paris", 1923, None),
        # Pixar Animation Studios (2006-; co-branded "Disney/Pixar")
        BrandCompany("Pixar Animation Studios", 2006, None),
        BrandCompany("Pixar", 2006, None),
        # Marvel Studios (2009-; MCU marketed as Disney IP)
        BrandCompany("Marvel Studios", 2009, None),
        # Lucasfilm Ltd. (2012-; Star Wars fully Disney-branded)
        BrandCompany("Lucasfilm", 2012, None),
        BrandCompany("Lucasfilm Animation", 2012, None),
        # DisneyToon Studios (1988-2018; Disney-branded DTV sequels)
        BrandCompany("Disneytoon Studios", 1988, 2018),
    ))

    # WALT_DISNEY_ANIMATION — the in-house theatrical animation studio
    # lineage. Only the two headline credits casual viewers associate with
    # the brand's canon (Little Mermaid → Frozen → Moana → Encanto).
    # Satellite facilities (Australia/Japan/Canada/France) and the
    # pre-1986 `Walt Disney Productions` credit (which mixed animation
    # with live-action) are excluded — they dilute brand purity.
    WALT_DISNEY_ANIMATION = ("walt-disney-animation", 2, "Walt Disney Animation Studios", (
        BrandCompany("Walt Disney Animation Studios", 2007, None),
        BrandCompany("Walt Disney Feature Animation", 1986, 2007),
    ))

    # PIXAR — single-identity studio, Emeryville CG animation.
    PIXAR = ("pixar", 3, "Pixar Animation Studios", (
        BrandCompany("Pixar Animation Studios", 1986, None),
        BrandCompany("Pixar", 1986, None),
    ))

    # MARVEL_STUDIOS — the MCU era. Pre-1996 `Marvel Films` predecessor
    # dropped: its output (unreleased Fantastic Four, late-80s TV movie
    # orbit) isn't what casual viewers mean by "Marvel Studios."
    MARVEL_STUDIOS = ("marvel-studios", 4, "Marvel Studios", (
        BrandCompany("Marvel Studios", 1996, None),
    ))

    # LUCASFILM — Star Wars + Indiana Jones. `Lucasfilm Ltd.` is the
    # formal IMDB credit. Animation arm added for Clone Wars and animated
    # output. ILM/Skywalker Sound excluded — VFX/sound, not production.
    LUCASFILM = ("lucasfilm", 5, "Lucasfilm", (
        BrandCompany("Lucasfilm", 1971, None),
        BrandCompany("Lucasfilm Ltd.", 1971, None),
        BrandCompany("Lucasfilm Animation", 2003, None),
    ))

    # WARNER_BROS — core Warner Bros. shield brand only. Excluded: New
    # Line (autonomous — LOTR, Rush Hour read as "New Line"), Castle
    # Rock, HBO Films (strong standalone brand), DC Studios (own brand),
    # Turner library. All covered by their own standalone registry
    # entries where relevant (NEW_LINE_CINEMA, DC).
    WARNER_BROS = ("warner-bros", 6, "Warner Bros.", (
        # Warner Bros. Pictures (1923-)
        BrandCompany("Warner Bros.", 1923, None),
        BrandCompany("Warner Bros. Entertainment", 1923, None),
        BrandCompany("Warner Brothers-First National Productions", 1923, None),
        BrandCompany("Warner Bros./Seven Arts", 1923, None),
        BrandCompany("Warner Bros. Pictures", 1923, None),
        BrandCompany("Warner Bros. First National", 1923, None),
        BrandCompany("Warner Bros Entertainment", 1923, None),
        BrandCompany("Warner Bros. Productions", 1923, None),
        BrandCompany("Warner Bros.-First National Pictures", 1923, None),
        BrandCompany("Warner Brothers Entertainment", 1923, None),
        BrandCompany("Warner Brothers Pictures", 1923, None),
        # Warner Bros. Animation (1980-)
        BrandCompany("Warner Bros. Cartoon Studios", 1980, None),
        BrandCompany("Warner Bros. Animation", 1980, None),
        BrandCompany("Warner Bros. Pictures Animation", 1980, None),
        BrandCompany("Warner Bros. Feature Animation", 1980, None),
        BrandCompany("Warner Classic Animation", 1980, None),
        BrandCompany("Warner Brothers/Seven Arts Animation", 1980, None),
        BrandCompany("Warner Bros. New York Animation", 1980, None),
    ))

    # NEW_LINE_CINEMA — core naming variants only. Heron JVs are obscure
    # coproduction shells (casual viewers have never heard of them);
    # Fine Line Features was the arthouse imprint, tonally distinct from
    # the LOTR/Rush Hour/Elm Street identity casual viewers expect.
    NEW_LINE_CINEMA = ("new-line-cinema", 7, "New Line Cinema", (
        BrandCompany("New Line Cinema", 1967, None),
        BrandCompany("New Line Productions", 1967, None),
        BrandCompany("New Line Film", 1967, None),
        BrandCompany("New Line Film Productions", 1967, None),
    ))

    # DC — live-action DCEU and current DC Studios production labels.
    # Pre-2009 `DC Comics` dropped: IMDB applies it broadly as a source
    # attribution across animated DTV and licensed tie-ins, which pulls
    # in content casual viewers don't mean by "DC movies." Classic
    # Burton/Schumacher Batman surfaces via Warner Bros. + franchise.
    DC = ("dc", 8, "DC Studios", (
        BrandCompany("DC Entertainment", 2009, 2016),
        BrandCompany("DC Films", 2016, 2022),
        BrandCompany("DC Studios", 2022, None),
    ))

    # UNIVERSAL — core Universal Pictures + Illumination (Universal
    # actively co-brands Minions/Despicable Me, theme-park integration).
    # Excluded: Focus Features (autonomous arthouse), DreamWorks
    # Animation (distinct household brand), Working Title (British
    # rom-com identity), Gramercy (PolyGram-era indie).
    UNIVERSAL = ("universal", 9, "Universal Pictures", (
        # Universal Pictures core
        BrandCompany("Universal Pictures", None, None),
        BrandCompany("Universal Film Manufacturing Company", None, None),
        BrandCompany("Universal International Pictures (UI)", None, None),
        BrandCompany("Universal Pictures International (UPI)", None, None),
        BrandCompany("Universal", None, None),
        # Illumination (2007-; co-branded "Universal & Illumination")
        BrandCompany("Illumination Entertainment", 2007, None),
        BrandCompany("Illumination", 2018, None),
        BrandCompany("Illumination Studios Paris", 2011, None),
    ))

    # FOCUS_FEATURES — prestige arthouse brand. Good Machine and USA
    # Films are merger precursors whose catalog matches the Focus
    # identity (Ice Storm, Traffic, Gosford Park, Being John Malkovich).
    # Focus World (niche VOD/genre) and Africa First (short-film
    # residency) excluded — casual viewers don't know them and they
    # dilute the prestige signal.
    FOCUS_FEATURES = ("focus-features", 10, "Focus Features", (
        BrandCompany("Focus Features", 2002, None),
        BrandCompany("Focus Features International (FFI)", 2002, None),
        BrandCompany("Good Machine", 1991, 2002),
        BrandCompany("Good Machine Films", 1991, 2002),
        BrandCompany("USA Films", 1999, 2002),
    ))

    # PARAMOUNT — core Paramount Pictures + Paramount-prefixed specialty
    # labels (Players, Animation, Vantage, Classics). The Paramount
    # prefix signals co-branding, satisfying the casual-viewer test.
    # Excluded: Nickelodeon (distinct kids TV brand), MTV Films (MTV
    # brand identity), revived Republic Pictures (own identity).
    PARAMOUNT = ("paramount", 11, "Paramount Pictures", (
        # Paramount Pictures core
        BrandCompany("Paramount Pictures", None, None),
        BrandCompany("Paramount", None, None),
        BrandCompany("Paramount British Pictures", None, None),
        BrandCompany("Paramount Films", None, None),
        # Paramount-prefixed specialty/sub-labels
        BrandCompany("Paramount Players", 2017, None),
        BrandCompany("Paramount Animation", 2011, None),
        BrandCompany("Paramount Animation Studios", 2011, None),
        BrandCompany("Paramount Vantage", 2006, 2013),
        BrandCompany("Paramount Classics", 2006, 2013),
    ))

    # SONY — Sony's mainstream theatrical identity: Columbia (Spider-Man,
    # Ghostbusters, Men in Black), TriStar (Jerry Maguire, Philadelphia),
    # Screen Gems (Resident Evil, Underworld), Sony Pictures Animation
    # (Spider-Verse, Hotel Transylvania). Excluded: Sony Pictures
    # Classics (distinct arthouse brand — Whiplash, Call Me By Your
    # Name); foreign-region Columbia labels; distribution-only credits;
    # low-profile imprints (Stage 6, Triumph, SPWA).
    SONY = ("sony", 12, "Sony Pictures", (
        # Columbia Pictures core (1989- under Sony)
        BrandCompany("Columbia Pictures", 1989, None),
        BrandCompany("Columbia Pictures Corporation", 1989, None),
        BrandCompany("Columbia Pictures Industries", 1989, None),
        BrandCompany("Columbia Films", 1989, None),
        BrandCompany("Columbia Productions", 1989, None),
        BrandCompany("Columbia", 1989, None),
        # TriStar Pictures (1989- under Sony)
        BrandCompany("TriStar Pictures", 1989, None),
        BrandCompany("Tri-Star Pictures", 1989, None),
        BrandCompany("TriStar Productions", 1989, None),
        BrandCompany("Tri Star Productions", 1989, None),
        BrandCompany("Tri Star", 1989, None),
        # Screen Gems (1999-; Sony-branded genre)
        BrandCompany("Screen Gems", 1999, None),
        # Sony Pictures Animation (2002-)
        BrandCompany("Sony Pictures Animation", 2002, None),
        # Sony Pictures umbrella credits — unconditional
        BrandCompany("Sony Pictures", None, None),
        BrandCompany("Sony Pictures Entertainment", None, None),
        BrandCompany("Sony Pictures Entertainment Company", None, None),
        BrandCompany("Sony Pictures Studios", None, None),
    ))

    # COLUMBIA — the flagship Columbia Pictures brand. Historic British
    # and generic-variant labels dropped (obscure B-pictures, no casual
    # recall). Joint-venture shells (Columbia-Delphi, Columbia-Thompson)
    # dropped — casual viewers attribute those films to plain "Columbia."
    COLUMBIA = ("columbia", 13, "Columbia Pictures", (
        BrandCompany("Columbia Pictures", 1924, None),
        BrandCompany("Columbia Pictures Corporation", 1924, 1968),
    ))

    # TWENTIETH_CENTURY — the 1935-onward 20th Century Fox / Studios
    # lineage casual viewers expect (Alien, Die Hard, Home Alone,
    # Titanic, X-Men, Avatar, Deadpool). Pre-1935 Fox Film Corporation
    # dropped: silent/early-talkie era is film-historian territory, not
    # casual mental model of the brand. Fox 2000 Pictures kept — Fight
    # Club, Cast Away, Walk the Line, Life of Pi were marketed as Fox.
    TWENTIETH_CENTURY = ("twentieth-century", 14, "20th Century Studios", (
        # Twentieth Century Fox (1935-2020)
        BrandCompany("Twentieth Century Fox", 1935, 2020),
        BrandCompany("Twentieth Century-Fox Productions", 1935, 2020),
        BrandCompany("Twentieth Century Fox Animation", 1935, 2020),
        BrandCompany("20th Century Pictures", 1935, 2020),
        BrandCompany("20th Century Fox", 1935, 2020),
        BrandCompany("Twentieth Century Productions", 1935, 2020),
        BrandCompany("Twentieth Century-Fox Studios, Hollywood", 1935, 2020),
        BrandCompany("20th Century Foss", 1935, 2020),
        # Fox 2000 Pictures (1994-2020)
        BrandCompany("Fox 2000 Pictures", 1994, 2020),
        # 20th Century Studios (2020-)
        BrandCompany("20th Century Studios", 2020, None),
        BrandCompany("Twentieth Century Animation", 2020, None),
    ))

    # SEARCHLIGHT — Fox Searchlight → Searchlight rename.
    SEARCHLIGHT = ("searchlight", 15, "Searchlight Pictures", (
        BrandCompany("Fox Searchlight Pictures", 1994, 2020),
        BrandCompany("Searchlight Pictures", 2020, None),
    ))

    # DREAMWORKS_ANIMATION — the Shrek/Kung Fu Panda/HTTYD/Trolls
    # animation catalog only. Live-action DreamWorks (Gladiator, American
    # Beauty, Transformers) is a separate entity and excluded.
    DREAMWORKS_ANIMATION = ("dreamworks-animation", 16, "DreamWorks Animation", (
        BrandCompany("DreamWorks Animation", 2004, None),
        BrandCompany("DreamWorks Animation SKG", 2004, 2015),
        BrandCompany("Pacific Data Images (PDI)", 1994, 2015),
        BrandCompany("PDI/DreamWorks", 1996, 2015),
    ))

    # ILLUMINATION — Despicable Me / Minions / Mario studio. Rebranded
    # from "Illumination Entertainment" to "Illumination" in 2018. Paris
    # animation arm kept (produces the flagship films). Mac Guff Ligne
    # dropped — its pre-2011 output is French VFX work casual viewers
    # don't associate with Illumination.
    ILLUMINATION = ("illumination", 17, "Illumination", (
        BrandCompany("Illumination Entertainment", 2007, 2018),
        BrandCompany("Illumination", 2018, None),
        BrandCompany("Illumination Studios Paris", 2011, None),
    ))

    # MGM — flagship Metro-Goldwyn-Mayer lion: classic Hollywood plus
    # Bond/Rocky/Hobbit/Creed. Tom and Jerry / Tex Avery shorts era
    # animation dropped (TV/shorts, not casual theatrical expectation).
    # MGM-Pathé two-year shell and MGM Producción (Spanish regional)
    # dropped. British Studios kept — made 2001: A Space Odyssey and
    # early Bond at Borehamwood.
    MGM = ("mgm", 18, "Metro-Goldwyn-Mayer", (
        BrandCompany("Metro-Goldwyn-Mayer (MGM)", 1924, None),
        BrandCompany("Metro-Goldwyn-Mayer (MGM) Studios", 1986, None),
        BrandCompany("Metro-Goldwyn-Mayer Studios", 1986, None),
        BrandCompany("Metro-Goldwyn-Mayer British Studios", 1936, 1970),
        BrandCompany("MGM British Studios", 1936, 1970),
    ))

    # LIONSGATE — Lionsgate naming variants + Summit post-acquisition.
    # Summit is post-2012 only, to exclude the Twilight-era competitor
    # years. Mandate and Artisan dropped — casual viewers remember their
    # signature catalogs (Juno, Blair Witch) as Mandate/Artisan indie,
    # not Lionsgate.
    LIONSGATE = ("lionsgate", 19, "Lionsgate", (
        BrandCompany("Lionsgate", None, None),
        BrandCompany("Lion's Gate Films", None, None),
        BrandCompany("Lions Gate Films", None, None),
        BrandCompany("Lions Gate Entertainment", None, None),
        BrandCompany("Lions Gate", None, None),
        BrandCompany("Lionsgate Premiere", None, None),
        BrandCompany("Lionsgate Productions", None, None),
        BrandCompany("Lions Gate Studios", None, None),
        BrandCompany("Summit Entertainment", 2012, None),
        BrandCompany("Summit Premiere", 2012, None),
    ))

    # A24 — prestige indie brand, primary surface plus common IMDB LLC
    # and "Films" variants.
    A24 = ("a24", 20, "A24", (
        BrandCompany("A24", None, None),
        BrandCompany("A24 Films", None, None),
        BrandCompany("A24 Films LLC", None, None),
    ))

    # NEON — primary "Neon" brand plus "Neon Rated" legal-entity form.
    NEON = ("neon", 21, "Neon", (
        BrandCompany("Neon", None, None),
        BrandCompany("Neon Rated", None, None),
    ))

    # BLUMHOUSE — Blumhouse Productions + International. Atomic Monster
    # gated to 2024 (post-merger); pre-merger Conjuring/Saw stays under
    # Atomic Monster's own identity. Blumhouse Television dropped —
    # "Blumhouse movies" query intent is theatrical.
    BLUMHOUSE = ("blumhouse", 22, "Blumhouse Productions", (
        BrandCompany("Blumhouse Productions", None, None),
        BrandCompany("Blumhouse International", None, None),
        BrandCompany("Atomic Monster", 2024, None),
    ))

    # STUDIO_GHIBLI — primary brand plus formal "Studio Ghibli, Inc."
    # legal-entity surface.
    STUDIO_GHIBLI = ("studio-ghibli", 23, "Studio Ghibli", (
        BrandCompany("Studio Ghibli", None, None),
        BrandCompany("Studio Ghibli, Inc.", None, None),
    ))

    # NETFLIX — producer-intent brand. "A Netflix Original Documentary"
    # dropped (marketing tagline, not a production_companies entry).
    NETFLIX = ("netflix", 24, "Netflix", (
        BrandCompany("Netflix", None, None),
        BrandCompany("Netflix Studios", None, None),
        BrandCompany("Netflix Animation", None, None),
        BrandCompany("Netflix Worldwide Entertainment", None, None),
        BrandCompany("Netflix Worldwide Productions", None, None),
        BrandCompany("Netflix India", None, None),
    ))

    # =====================================================================
    # Tier 2 — 7 sub-labels / streamer-producers still in MVP
    # =====================================================================

    # SONY_PICTURES_ANIMATION — single-member standalone (also in SONY).
    SONY_PICTURES_ANIMATION = ("sony-pictures-animation", 25, "Sony Pictures Animation", (
        BrandCompany("Sony Pictures Animation", 2002, None),
    ))

    # TRISTAR — canonical variants + 2015 revival label. Space-separated
    # `Tri Star` was a data-entry artifact, dropped.
    TRISTAR = ("tristar", 26, "TriStar Pictures", (
        BrandCompany("TriStar Pictures", 1982, None),
        BrandCompany("Tri-Star Pictures", 1982, 1991),
        BrandCompany("TriStar Productions", 2015, None),
    ))

    # TOUCHSTONE — label dormant after 2018 but retained for its
    # historical catalog (Pretty Woman, Sister Act, Sixth Sense).
    TOUCHSTONE = ("touchstone", 27, "Touchstone Pictures", (
        BrandCompany("Touchstone Pictures", 1986, 2018),
        BrandCompany("Touchstone Films", 1984, 1986),
    ))

    # MIRAMAX — Weinstein-era prestige indie catalog (Pulp Fiction, Good
    # Will Hunting, Kill Bill, No Country). Dimension Films dropped —
    # horror/family-genre (Scream, Spy Kids, Scary Movie) built its own
    # identity distinct from Miramax's prestige brand.
    MIRAMAX = ("miramax", 28, "Miramax", (
        BrandCompany("Miramax", 1979, None),
        BrandCompany("Miramax Films", 1979, None),
    ))

    # UNITED_ARTISTS — two active eras with a dormant gap. `United
    # Artists Pictures` covers the 1981-2018 MGM-era releasing label.
    # Corporate/regional-distribution strings dropped (Europa, Film
    # Corporation) — not casual-facing credits.
    UNITED_ARTISTS = ("united-artists", 29, "United Artists", (
        BrandCompany("United Artists", 1919, 1981),
        BrandCompany("United Artists", 2024, None),
        BrandCompany("United Artists Pictures", 1981, 2018),
    ))

    # AMAZON_MGM — additive umbrella for post-acquisition MGM and UA
    # output. Sub-labels (MGM Animation, Family, British Studios,
    # Producción; UA Film Corporation, Europa; Amazon Studios Germany)
    # dropped — none are Amazon-era casual-viewer credits.
    AMAZON_MGM = ("amazon-mgm", 30, "Amazon MGM Studios", (
        BrandCompany("Amazon Studios", None, None),
        BrandCompany("Amazon MGM Studios", None, None),
        BrandCompany("Metro-Goldwyn-Mayer (MGM)", 2022, None),
        BrandCompany("Metro-Goldwyn-Mayer (MGM) Studios", 2022, None),
        BrandCompany("Metro-Goldwyn-Mayer Studios", 2022, None),
        BrandCompany("United Artists", 2024, None),
        BrandCompany("United Artists Pictures", 2024, None),
    ))

    # APPLE_STUDIOS — Apple Original Films + Apple Studios (in-house).
    APPLE_STUDIOS = ("apple-studios", 31, "Apple Studios", (
        BrandCompany("Apple Original Films", None, None),
        BrandCompany("Apple Studios", None, None),
    ))


# ---------------------------------------------------------------------------
# Reverse index: surface string → [(brand, start_year, end_year), ...]
#
# Sorted alphabetically by brand.name (ascending), then by start_year
# (ascending, None as -inf), then by end_year (ascending, None as +inf).
# This gives callers a deterministic iteration sequence.
# ---------------------------------------------------------------------------

_MembershipRow = tuple[ProductionBrand, int | None, int | None]


def _build_and_validate_registry() -> dict[str, list[_MembershipRow]]:
    """Construct the reverse index and validate registry invariants.

    Runs exactly once at import. Asserts brand_id uniqueness, non-empty
    display names, at least one company per brand, non-empty surface
    strings, and sane year windows. Returns the populated reverse index
    with deterministic ordering per the module docstring.
    """
    index: dict[str, list[_MembershipRow]] = {}
    seen_ids: set[int] = set()
    for brand in ProductionBrand:
        assert brand.brand_id not in seen_ids, (
            f"Duplicate brand_id {brand.brand_id} on {brand.name}"
        )
        seen_ids.add(brand.brand_id)
        assert brand.display_name, f"{brand.name} has empty display_name"
        assert brand.companies, f"{brand.name} has empty companies tuple"
        for company in brand.companies:
            assert company.string, (
                f"{brand.name} has a BrandCompany with empty string"
            )
            if company.start_year is not None and company.end_year is not None:
                assert company.start_year <= company.end_year, (
                    f"{brand.name} / {company.string!r}: "
                    f"start_year {company.start_year} > end_year {company.end_year}"
                )
            index.setdefault(company.string, []).append(
                (brand, company.start_year, company.end_year)
            )

    def _sort_key(row: _MembershipRow) -> tuple[str, float, float]:
        row_brand, start, end = row
        return (
            row_brand.name,
            float("-inf") if start is None else float(start),
            float("inf") if end is None else float(end),
        )

    for rows in index.values():
        rows.sort(key=_sort_key)
    return index


_STRING_TO_MEMBERSHIPS: dict[str, list[_MembershipRow]] = _build_and_validate_registry()


def memberships_for_string(s: str) -> list[_MembershipRow]:
    """Return every (brand, start_year, end_year) row that lists `s`.

    Returns an empty list for strings not in the registry — callers apply
    year-gating on top.
    """
    return _STRING_TO_MEMBERSHIPS.get(s, [])


def year_matches(
    start: int | None,
    end: int | None,
    release_year: int | None,
) -> bool:
    """Is `release_year` inside the [start, end] window?

    Both bounds inclusive. None means "no bound on that side".

    Special rule for release_year=None: only unconditional memberships
    (start=None and end=None) match — any windowed row is skipped. This
    matches the ingestion-time directive: a movie with an unknown
    release year cannot be confidently bound to a time-gated brand.
    """
    if release_year is None:
        return start is None and end is None
    if start is not None and release_year < start:
        return False
    if end is not None and release_year > end:
        return False
    return True
