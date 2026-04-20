# Production-brand registry for the MVP studio resolver.
#
# Backstory: `search_improvement_planning/production_company_tiers.md` finalized
# 31 studio brands with their member production companies, each company's exact
# IMDB surface strings, and the time window during which that company was part
# of the brand. This module encodes that registry in a form the ingestion
# pipeline can consume without touching the markdown file.
#
# Shape:
#   - `ProductionBrand` is a str-backed enum; the enum value is the brand slug
#     (e.g. "disney", "walt-disney-animation"). Each member also carries
#     `brand_id` (int) and `display_name` (str) attributes.
#   - `ProductionBrand.companies` is a tuple of `BrandCompany` rows. Each row
#     is an exact IMDB `production_companies` string plus the (start_year,
#     end_year) window during which that string SHOULD count as this brand.
#
# Year conventions (authoritative copy is
# `unit_tests/production_brand_spec_dates.py::EXPECTED_DATE_MEMBERSHIPS`):
#   - start_year / end_year are **inclusive**. `None` means "no bound".
#   - `(None, None)` = "always applicable for this brand". Used only for
#     strings that can't be pinned to a founding/acquisition date (umbrella
#     credit strings like "Sony Pictures Entertainment", bare "A24", etc.).
#   - Every other row carries the member's historical or acquisition window
#     per the tier doc.
#   - A single surface string can appear under multiple brands with different
#     windows (e.g. `Miramax` → MIRAMAX 1979-, DISNEY 1993-2010). The resolver
#     emits every brand that passes the year check.
#   - A single surface string may also appear TWICE within one brand if that
#     brand had non-overlapping active eras (e.g. UA's `United Artists` has
#     both 1919-1981 and 2024- rows). The resolver handles this correctly —
#     each entry independently passes or fails the year check.
#
# Lookup:
#   - `memberships_for_string(s)` returns `[(ProductionBrand, start_year,
#     end_year), ...]` sorted alphabetically by brand name (ascending), then
#     by start_year (ascending, None as -inf), then by end_year. This order
#     matches `EXPECTED_DATE_MEMBERSHIPS` for parametric test assertions.
#   - `year_matches(start, end, release_year)` encodes the window predicate.
#     If release_year is None, only `(None, None)` rows match — any window is
#     skipped per the user's ingestion-time rule.
#
# Ingestion side wires this into `movie_ingestion/final_ingestion/
# brand_resolver.py`, which takes an IMDB `production_companies` list + a
# release year and returns a list of `BrandTag(brand_id, first_matching_index)`.

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class BrandCompany:
    """One surface-string → brand relationship with its time window.

    `string` is the exact IMDB `production_companies` value as it appears in
    the database (case- and whitespace-sensitive). Matching uses exact string
    comparison at this stage; any normalization layer would sit on top.

    `start_year` / `end_year` are inclusive and open-ended when None. See the
    module docstring for the "always applicable" (None, None) convention.
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
        # Enum value is the slug (string); the other fields are attached as
        # attributes so each member is a hashable string carrying extra data.
        obj = str.__new__(cls, slug)
        obj._value_ = slug
        obj.brand_id = brand_id
        obj.display_name = display_name
        obj.companies = companies
        return obj

    # =====================================================================
    # Tier 1 — 24 major brands
    # =====================================================================

    # DISNEY — umbrella for the full Walt Disney Studios catalog, including
    # the post-2019 Fox assets. All sub-brand surfaces are included here with
    # Disney-era acquisition windows so Disney tagging works for films whose
    # only production credit is the acquired sub-brand.
    DISNEY = ("disney", 1, "The Walt Disney Studios", (
        # Walt Disney Pictures (1983-)
        BrandCompany("Walt Disney Pictures", 1983, None),
        BrandCompany("Walt Disney Studios", 1983, None),
        BrandCompany("Walt Disney Studios Motion Pictures", 1983, None),
        BrandCompany("Walt Disney Home Video", 1983, None),
        BrandCompany("Walt Disney Home Entertainment", 1983, None),
        BrandCompany("Walt Disney British Films", 1983, None),
        BrandCompany("Walt Disney Studios Home Entertainment", 1983, None),
        BrandCompany("Walt Disney Pictures / Sony Pictures", 1983, None),
        # Walt Disney Productions (1929-1986)
        BrandCompany("Walt Disney Productions", 1929, 1986),
        # Walt Disney Animation Studios lineage rolled into DISNEY (1923-)
        BrandCompany("Walt Disney Animation Studios", 1923, None),
        BrandCompany("Walt Disney Feature Animation", 1923, None),
        BrandCompany("Walt Disney Animation Australia", 1923, None),
        BrandCompany("Walt Disney Animation Japan", 1923, None),
        BrandCompany("Walt Disney Feature Animation Florida", 1923, None),
        BrandCompany("Walt Disney Animation Canada", 1923, None),
        BrandCompany("Walt Disney Animation France S.A.", 1923, None),
        BrandCompany("Walt Disney Feature Animation Paris", 1923, None),
        # Pixar Animation Studios (2006-)
        BrandCompany("Pixar Animation Studios", 2006, None),
        BrandCompany("Pixar", 2006, None),
        # Marvel Studios (2009-)
        BrandCompany("Marvel Studios", 2009, None),
        # Lucasfilm Ltd. (2012-)
        BrandCompany("Lucasfilm", 2012, None),
        BrandCompany("Lucasfilm Animation", 2012, None),
        # 20th Century Studios (2019-)
        BrandCompany("20th Century Studios", 2019, None),
        BrandCompany("20th Century Fox Home Entertainment", 2019, None),
        BrandCompany("20th Century Fox Argentina", 2019, None),
        BrandCompany("20th Century Fox Korea", 2019, None),
        BrandCompany("20th Century Fox Post Production Services", 2019, None),
        # 20th Century Fox (2019-2020 under Disney, pre-rename)
        BrandCompany("Twentieth Century Fox", 2019, 2020),
        BrandCompany("Twentieth Century-Fox Productions", 2019, 2020),
        BrandCompany("Twentieth Century Fox Animation", 2019, 2020),
        BrandCompany("20th Century Pictures", 2019, 2020),
        BrandCompany("20th Century Fox", 2019, 2020),
        BrandCompany("Twentieth Century Animation", 2019, 2020),
        BrandCompany("Twentieth Century Productions", 2019, 2020),
        BrandCompany("Twentieth Century-Fox Studios, Hollywood", 2019, 2020),
        # Searchlight Pictures (2019-)
        BrandCompany("Searchlight Pictures", 2019, None),
        # Fox Searchlight Pictures (2019-2020)
        BrandCompany("Fox Searchlight Pictures", 2019, 2020),
        # Touchstone Pictures (1984-2018)
        BrandCompany("Touchstone Pictures", 1984, 2018),
        BrandCompany("Touchstone Films", 1984, 2018),
        BrandCompany("Touchstone", 1984, 2018),
        BrandCompany("Touchstone Pictures México", 1984, 2018),
        # Hollywood Pictures (1989-2007)
        BrandCompany("Hollywood Pictures", 1989, 2007),
        BrandCompany("Hollywood Pictures Corporation (I)", 1989, 2007),
        BrandCompany("Hollywood Pictures Corporation (II)", 1989, 2007),
        BrandCompany("Hollywood Pictures Home Video", 1989, 2007),
        # Miramax Films (1993-2010)
        BrandCompany("Miramax", 1993, 2010),
        BrandCompany("Miramax Family Films", 1993, 2010),
        BrandCompany("Miramax International", 1993, 2010),
        BrandCompany("Miramax Home Entertainment", 1993, 2010),
        # Dimension Films (1993-2005)
        BrandCompany("Dimension Films", 1993, 2005),
        BrandCompany("Dimension Films (II)", 1993, 2005),
        # DisneyToon Studios (1988-2018, start corrected via Wikidata)
        BrandCompany("Disneytoon Studios", 1988, 2018),
        # Blue Sky Studios (2019-2021)
        BrandCompany("Blue Sky Studios", 2019, 2021),
    ))

    # WALT_DISNEY_ANIMATION — the in-house theatrical animation studio lineage
    # (distinct from DisneyToon/Pixar/Fox animation). Three era-gated member
    # tiers: current name (2007-), prior Feature Animation name (1986-2007),
    # and pre-1986 parent-company credit on classic-era features.
    WALT_DISNEY_ANIMATION = ("walt-disney-animation", 2, "Walt Disney Animation Studios", (
        # Walt Disney Animation Studios (2007-)
        BrandCompany("Walt Disney Animation Studios", 2007, None),
        BrandCompany("Walt Disney Animation Australia", 2007, None),
        BrandCompany("Walt Disney Animation Japan", 2007, None),
        BrandCompany("Walt Disney Animation Canada", 2007, None),
        BrandCompany("Walt Disney Animation France S.A.", 2007, None),
        # Walt Disney Feature Animation (1986-2007)
        BrandCompany("Walt Disney Feature Animation", 1986, 2007),
        BrandCompany("Walt Disney Feature Animation Florida", 1986, 2007),
        BrandCompany("Walt Disney Feature Animation Paris", 1986, 2007),
        # Walt Disney Productions (1929-1986)
        BrandCompany("Walt Disney Productions", 1929, 1986),
    ))

    # PIXAR — single member (the founding company), gated to its founding year.
    PIXAR = ("pixar", 3, "Pixar Animation Studios", (
        BrandCompany("Pixar Animation Studios", 1986, None),
        BrandCompany("Pixar", 1986, None),
    ))

    # MARVEL_STUDIOS — Marvel Studios plus the pre-1996 predecessor name.
    MARVEL_STUDIOS = ("marvel-studios", 4, "Marvel Studios", (
        BrandCompany("Marvel Studios", 1996, None),
        BrandCompany("Marvel Films", 1993, 1996),
    ))

    # LUCASFILM — founded 1971.
    LUCASFILM = ("lucasfilm", 5, "Lucasfilm", (
        BrandCompany("Lucasfilm", 1971, None),
    ))

    # WARNER_BROS — umbrella including New Line, Castle Rock, Turner, HBO,
    # and the DC film labels. DC Comics (pre-2009 publisher credit) included
    # per tier-doc member row.
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
        # New Line Cinema (under WB: 1996-)
        BrandCompany("New Line Cinema", 1996, None),
        BrandCompany("Fine Line Features", 1996, None),
        BrandCompany("New Line Productions", 1996, None),
        BrandCompany("New Line Film", 1996, None),
        BrandCompany("New Line Film Productions", 1996, None),
        # Castle Rock Entertainment (1996-)
        BrandCompany("Castle Rock Entertainment", 1996, None),
        # Turner Pictures (1996-1998)
        BrandCompany("Turner Pictures (I)", 1996, 1998),
        BrandCompany("Turner Pictures Worldwide", 1996, 1998),
        BrandCompany("Ted Turner Pictures", 1996, 1998),
        BrandCompany("Turner Pictures (III)", 1996, 1998),
        # HBO Films (1999-)
        BrandCompany("Home Box Office (HBO)", 1999, None),
        BrandCompany("HBO Documentary Films", 1999, None),
        BrandCompany("HBO Films", 1999, None),
        BrandCompany("HBO Max", 1999, None),
        BrandCompany("HBO Premiere Films", 1999, None),
        # HBO Pictures (1989-1999)
        BrandCompany("HBO Pictures", 1989, 1999),
        # DC Entertainment (2009-2022)
        BrandCompany("DC Entertainment", 2009, 2022),
        # DC Films (2016-2022)
        BrandCompany("DC Films", 2016, 2022),
        # DC Studios (2022-)
        BrandCompany("DC Studios", 2022, None),
        # DC Comics publisher credit on pre-2009 WB DC films (open curation
        # item 1 in the tier doc — retained per the member row).
        BrandCompany("DC Comics", None, 2009),
    ))

    # NEW_LINE_CINEMA — New Line Cinema (1967-) + Fine Line Features
    # imprint. Joint-venture entities are unconditional — only two rows
    # each with no clear date assertion.
    NEW_LINE_CINEMA = ("new-line-cinema", 7, "New Line Cinema", (
        BrandCompany("New Line Cinema", 1967, None),
        BrandCompany("New Line Productions", 1967, None),
        BrandCompany("New Line Film", 1967, None),
        BrandCompany("New Line Film Productions", 1967, None),
        BrandCompany("The New Line-Heron Joint Venture", None, None),
        BrandCompany("The Fourth New Line-Heron Joint Venture", None, None),
        BrandCompany("Fine Line Features", 1990, 2005),
    ))

    # DC — DC Studios lineage. Pre-2009 DC Comics credit retained per tier
    # doc open curation item.
    DC = ("dc", 8, "DC Studios", (
        BrandCompany("DC Comics", None, 2009),
        BrandCompany("DC Entertainment", 2009, 2016),
        BrandCompany("DC Films", 2016, 2022),
        BrandCompany("DC Studios", 2022, None),
    ))

    # UNIVERSAL — umbrella. Focus / Illumination / DreamWorks / Working Title
    # windows start at the Universal acquisition date, NOT at the sub-label's
    # founding. A film pre-dating acquisition and credited only to the
    # sub-label is tagged to the standalone brand, not UNIVERSAL.
    UNIVERSAL = ("universal", 9, "Universal Pictures", (
        # Universal Pictures core — unconditional umbrella strings
        BrandCompany("Universal Pictures", None, None),
        BrandCompany("Universal Film Manufacturing Company", None, None),
        BrandCompany("Universal International Pictures (UI)", None, None),
        BrandCompany("Universal Pictures International (UPI)", None, None),
        BrandCompany("Universal", None, None),
        # Focus Features (2002-)
        BrandCompany("Focus Features", 2002, None),
        BrandCompany("Focus Features International (FFI)", 2002, None),
        BrandCompany("Focus World", 2002, None),
        BrandCompany("Focus Features Africa First Program", 2002, None),
        # Focus Features predecessors — under UNIVERSAL umbrella from
        # Focus's formation (2002). Spec intentionally uses the umbrella
        # acquisition year, not the predecessor's own historical window.
        BrandCompany("USA Films", 2002, None),
        BrandCompany("Good Machine", 2002, None),
        BrandCompany("Good Machine Films", 2002, None),
        # Illumination (2007-)
        BrandCompany("Illumination Entertainment", 2007, None),
        # Illumination Mac Guff (2011-)
        BrandCompany("Mac Guff Ligne", 2011, None),
        # DreamWorks Animation (2016-)
        BrandCompany("DreamWorks Animation", 2016, None),
        BrandCompany("Pacific Data Images (PDI)", 2016, None),
        # Working Title Films (1999-)
        BrandCompany("Working Title Films", 1999, None),
        BrandCompany("WT2 Productions", 1999, None),
        # Gramercy Pictures (1992-1999)
        BrandCompany("Gramercy Pictures (I)", 1992, 1999),
        BrandCompany("Gramercy Pictures (II)", 1992, 1999),
    ))

    # FOCUS_FEATURES — its own brand (also member of UNIVERSAL always,
    # encoded above with acquisition year).
    FOCUS_FEATURES = ("focus-features", 10, "Focus Features", (
        BrandCompany("Focus Features", 2002, None),
        BrandCompany("Focus Features International (FFI)", 2002, None),
        BrandCompany("Focus Features Africa First Program", 2002, None),
        BrandCompany("Focus World", 2010, 2013),
        BrandCompany("Good Machine", 1991, 2002),
        BrandCompany("Good Machine Films", 1991, 2002),
        BrandCompany("USA Films", 1999, 2002),
    ))

    # PARAMOUNT — umbrella. Republic Pictures surface is year-gated to the
    # 2023 revival per tier-doc note; the same string covers the 1935-1967
    # legacy company and must NOT tag pre-2023 films.
    PARAMOUNT = ("paramount", 11, "Paramount Pictures", (
        # Paramount Pictures core — unconditional umbrella strings
        BrandCompany("Paramount Pictures", None, None),
        BrandCompany("Paramount", None, None),
        BrandCompany("Paramount British Pictures", None, None),
        BrandCompany("Paramount Films", None, None),
        # Paramount Players (2017-)
        BrandCompany("Paramount Players", 2017, None),
        # Paramount Animation (2011-)
        BrandCompany("Paramount Animation", 2011, None),
        BrandCompany("Paramount Animation Studios", 2011, None),
        # Paramount Vantage (2006-2013)
        BrandCompany("Paramount Vantage", 2006, 2013),
        BrandCompany("Paramount Classics", 2006, 2013),
        # Nickelodeon Movies (1995-)
        BrandCompany("Nickelodeon Movies", 1995, None),
        BrandCompany("Nickelodeon Animation Studios", 1995, None),
        BrandCompany("Nickelodeon Films", 1995, None),
        # MTV Films (1996-)
        BrandCompany("MTV Films", 1996, None),
        BrandCompany("MTV Entertainment Studios", 1996, None),
        BrandCompany("MTV Films Europe", 1996, None),
        # Republic Pictures (2023- revival; STRICTLY year-gated)
        BrandCompany("Republic Pictures", 2023, None),
        BrandCompany("Republic Pictures (III)", 2023, None),
    ))

    # SONY — umbrella spanning Columbia, TriStar, Screen Gems, SPA, and
    # related imprints. All sub-brand rows start at Sony's acquisition year
    # (or later, for post-acquisition foundings). Umbrella credits (Sony
    # Pictures Entertainment etc.) included as always-applicable.
    SONY = ("sony", 12, "Sony Pictures", (
        # Columbia Pictures (1989- under Sony)
        BrandCompany("Columbia Pictures", 1989, None),
        BrandCompany("Columbia Pictures Corporation", 1989, None),
        BrandCompany("Columbia Pictures Entertainment", 1989, None),
        BrandCompany("Columbia Pictures Industries", 1989, None),
        BrandCompany("Columbia Pictures Film Production Asia", 1989, None),
        BrandCompany("Deutsche Columbia Pictures Film Produktion", 1989, None),
        BrandCompany("Columbia Pictures do Brasil", 1989, None),
        BrandCompany("Columbia Pictures of Brasil", 1989, None),
        BrandCompany("Columbia Pictures Producciones Mexico", 1989, None),
        BrandCompany("Columbia Pictures of Argentina", 1989, None),
        BrandCompany("Columbia Films", 1989, None),
        BrandCompany("Columbia Films Productions", 1989, None),
        BrandCompany("Columbia Productions", 1989, None),
        BrandCompany("Columbia British Productions", 1989, None),
        BrandCompany("Columbia Release", 1989, None),
        BrandCompany("Columbia", 1989, None),
        # TriStar Pictures (1989- under Sony)
        BrandCompany("TriStar Pictures", 1989, None),
        BrandCompany("Tri-Star Pictures", 1989, None),
        BrandCompany("TriStar Productions", 1989, None),
        BrandCompany("Tri Star Productions", 1989, None),
        BrandCompany("Tri Star", 1989, None),
        # Screen Gems (1999-)
        BrandCompany("Screen Gems", 1999, None),
        # Sony Pictures Classics (1992-)
        BrandCompany("Sony Pictures Classics", 1992, None),
        # Sony Pictures Animation (2002-)
        BrandCompany("Sony Pictures Animation", 2002, None),
        # Stage 6 Films (2007-)
        BrandCompany("Stage 6 Films", 2007, None),
        BrandCompany("Stage 6 Productions", 2007, None),
        # Triumph Films (1989-)
        BrandCompany("Triumph Films", 1989, None),
        BrandCompany("Triumph Films (II)", 1989, None),
        # Sony Pictures umbrella credit — unconditional
        BrandCompany("Sony Pictures", None, None),
        BrandCompany("Sony Pictures Entertainment", None, None),
        BrandCompany("Sony Pictures Entertainment Company", None, None),
        BrandCompany("Sony Pictures Releasing", None, None),
        BrandCompany("Sony Pictures Releasing International", None, None),
        BrandCompany("Sony Pictures International", None, None),
        BrandCompany("Sony Pictures International Productions", None, None),
        BrandCompany("Sony Pictures Worldwide Acquisitions (SPWA)", None, None),
        BrandCompany("Sony Pictures Studios", None, None),
        BrandCompany("Sony BMG Feature Films", None, None),
        BrandCompany("Sony International Motion Picture Production Group", None, None),
        BrandCompany("Sony Pictures Films India", None, None),
        BrandCompany("Sony / Monumental Pictures", None, None),
    ))

    # COLUMBIA — its own brand (also member of SONY since 1989).
    COLUMBIA = ("columbia", 13, "Columbia Pictures", (
        BrandCompany("Columbia Pictures", 1924, None),
        BrandCompany("Columbia British Productions", 1924, None),
        BrandCompany("Columbia Films", 1924, None),
        BrandCompany("Columbia Productions", 1924, None),
        BrandCompany("Columbia Pictures Corporation", 1924, 1968),
        # Joint-venture strings with no clear date range — unconditional
        BrandCompany("Columbia-Delphi Productions", None, None),
        BrandCompany("Columbia-Thompson Venture", None, None),
    ))

    # TWENTIETH_CENTURY — multi-era brand lineage. Uses tier-doc member
    # windows. Fox Film Corporation-era and post-2020 eras included.
    TWENTIETH_CENTURY = ("twentieth-century", 14, "20th Century Studios", (
        # Twentieth Century Fox Film Corporation (1935-2020)
        BrandCompany("Twentieth Century Fox", 1935, 2020),
        BrandCompany("Twentieth Century-Fox Productions", 1935, 2020),
        BrandCompany("Twentieth Century Fox Animation", 1935, 2020),
        BrandCompany("20th Century Pictures", 1935, 2020),
        BrandCompany("20th Century Fox", 1935, 2020),
        BrandCompany("Twentieth Century Productions", 1935, 2020),
        BrandCompany("Twentieth Century-Fox Studios, Hollywood", 1935, 2020),
        BrandCompany("20th Century Foss", 1935, 2020),
        # Fox Film Corporation (pre-1935)
        BrandCompany("Fox Film Corporation", None, 1934),
        BrandCompany("Fox Film Company", None, 1934),
        BrandCompany("Fox Films", None, 1934),
        # Fox 2000 Pictures (1994-2020)
        BrandCompany("Fox 2000 Pictures", 1994, 2020),
        # 20th Century Studios (2020-)
        BrandCompany("20th Century Studios", 2020, None),
        BrandCompany("Twentieth Century Animation", 2020, None),
    ))

    # SEARCHLIGHT — two-era brand (Fox Searchlight → Searchlight).
    SEARCHLIGHT = ("searchlight", 15, "Searchlight Pictures", (
        BrandCompany("Fox Searchlight Pictures", 1994, 2020),
        BrandCompany("Searchlight Pictures", 2020, None),
    ))

    # DREAMWORKS_ANIMATION — standalone brand (also member of UNIVERSAL from
    # 2016, encoded under UNIVERSAL above).
    DREAMWORKS_ANIMATION = ("dreamworks-animation", 16, "DreamWorks Animation", (
        BrandCompany("DreamWorks Animation", 2004, None),
        BrandCompany("Pacific Data Images (PDI)", 1994, 2015),
    ))

    # ILLUMINATION — Illumination Entertainment became "Illumination" ~2018;
    # the Entertainment surface string only covers the legacy era.
    ILLUMINATION = ("illumination", 17, "Illumination", (
        BrandCompany("Illumination Entertainment", 2007, 2018),
        BrandCompany("Illumination Studios Paris", 2011, None),
        BrandCompany("Mac Guff Ligne", 2011, None),
    ))

    # MGM — legacy MGM catalog. Self-membership stays open-ended per user's
    # resolved convention; AMAZON_MGM picks up the post-2022 overlap
    # separately (additive, not migration).
    MGM = ("mgm", 18, "Metro-Goldwyn-Mayer", (
        BrandCompany("Metro-Goldwyn-Mayer (MGM)", 1924, None),
        BrandCompany("Metro-Goldwyn-Mayer Cartoon Studios", 1937, 1957),
        BrandCompany("Metro-Goldwyn-Mayer British Studios", 1936, 1970),
        BrandCompany("Metro-Goldwyn-Mayer Animation", 1937, 1957),
        BrandCompany("Metro-Goldwyn-Mayer (MGM) Studios", 1986, None),
        BrandCompany("Metro-Goldwyn-Mayer Studios", 1986, None),
        BrandCompany("MGM Animation/Visual Arts", 1937, 1957),
        BrandCompany("MGM Family Entertainment", 1986, None),
        BrandCompany("MGM British Studios", 1936, 1970),
        BrandCompany("MGM-Pathé Communications Co.", 1990, 1992),
        BrandCompany("MGM Producción", 1986, None),
    ))

    # LIONSGATE — includes Summit, Mandate, Artisan acquisitions. Lionsgate's
    # own name variants stay unconditional because their founding/rename
    # history isn't date-asserted in the tier doc.
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
        BrandCompany("Mandate Pictures", 2007, None),
        BrandCompany("Artisan Entertainment", 2003, None),
    ))

    # A24 — bare "A24" string, unconditional (no founding-year gate).
    A24 = ("a24", 20, "A24", (
        BrandCompany("A24", None, None),
    ))

    # NEON — bare "Neon" string, unconditional.
    NEON = ("neon", 21, "Neon", (
        BrandCompany("Neon", None, None),
    ))

    # BLUMHOUSE — Blumhouse Productions + Atomic Monster merger (2024).
    BLUMHOUSE = ("blumhouse", 22, "Blumhouse Productions", (
        BrandCompany("Blumhouse Productions", None, None),
        BrandCompany("Blumhouse International", None, None),
        BrandCompany("Blumhouse Television", 2008, None),
        BrandCompany("Atomic Monster", 2024, None),
    ))

    # STUDIO_GHIBLI — bare "Studio Ghibli" string.
    STUDIO_GHIBLI = ("studio-ghibli", 23, "Studio Ghibli", (
        BrandCompany("Studio Ghibli", None, None),
    ))

    # NETFLIX — producer-intent brand (platform intent handled by
    # watch_providers path per tier-doc streamer-disambiguation rule).
    NETFLIX = ("netflix", 24, "Netflix", (
        BrandCompany("Netflix", None, None),
        BrandCompany("Netflix Studios", None, None),
        BrandCompany("Netflix Animation", None, None),
        BrandCompany("A Netflix Original Documentary", None, None),
        BrandCompany("Netflix Worldwide Entertainment", None, None),
        BrandCompany("Netflix Worldwide Productions", None, None),
        BrandCompany("Netflix India", None, None),
    ))

    # =====================================================================
    # Tier 2 — 7 sub-labels / streamer-producers still in MVP
    # =====================================================================

    # SONY_PICTURES_ANIMATION — single-member standalone; also member of SONY.
    SONY_PICTURES_ANIMATION = ("sony-pictures-animation", 25, "Sony Pictures Animation", (
        BrandCompany("Sony Pictures Animation", 2002, None),
    ))

    # TRISTAR — its own brand, also member of SONY since 1989.
    TRISTAR = ("tristar", 26, "TriStar Pictures", (
        BrandCompany("TriStar Pictures", 1982, None),
        BrandCompany("Tri Star", 1982, None),
        BrandCompany("Tri-Star Pictures", 1982, 1991),
        BrandCompany("TriStar Productions", 2015, None),
    ))

    # TOUCHSTONE — label dormant after 2018 but retained.
    TOUCHSTONE = ("touchstone", 27, "Touchstone Pictures", (
        BrandCompany("Touchstone Pictures", 1986, 2018),
        BrandCompany("Touchstone Films", 1984, 1986),
    ))

    # MIRAMAX — spans multiple ownership eras. Dimension Films 1992-2005
    # window applies to THIS brand specifically (post-2005 Dimension belongs
    # to the Weinstein Company, out of MVP scope).
    MIRAMAX = ("miramax", 28, "Miramax", (
        BrandCompany("Miramax", 1979, None),
        BrandCompany("Dimension Films", 1992, 2005),
    ))

    # UNITED_ARTISTS — two active eras with a dormant gap. "United Artists"
    # appears TWICE with non-overlapping windows (1919-1981 classic,
    # 2024- revival). The intermediate MGM-era pictures (1981-2018) carry
    # distinct surface strings.
    UNITED_ARTISTS = ("united-artists", 29, "United Artists", (
        BrandCompany("United Artists", 1919, 1981),
        BrandCompany("United Artists", 2024, None),
        BrandCompany("United Artists Pictures", 1981, 2018),
        BrandCompany("United Artists Film Corporation", 1981, 2018),
        BrandCompany("United Artists Europa", 1981, 2018),
    ))

    # AMAZON_MGM — additive umbrella. Post-2022 MGM and post-2024 UA films
    # carry both the legacy standalone brand AND this umbrella per the tier
    # doc cross-brand membership table. Only the MGM sub-labels that were
    # still active at the Amazon acquisition (2022) are included — defunct
    # strings like Cartoon Studios/British Studios/Pathé predate Amazon era.
    AMAZON_MGM = ("amazon-mgm", 30, "Amazon MGM Studios", (
        # Amazon Studios era — unconditional
        BrandCompany("Amazon Studios", None, None),
        BrandCompany("Amazon MGM Studios", None, None),
        BrandCompany("Amazon Studios Germany", None, None),
        # MGM absorbed 2022 — only live MGM surfaces
        BrandCompany("Metro-Goldwyn-Mayer (MGM)", 2022, None),
        BrandCompany("Metro-Goldwyn-Mayer (MGM) Studios", 2022, None),
        BrandCompany("Metro-Goldwyn-Mayer Studios", 2022, None),
        BrandCompany("Metro-Goldwyn-Mayer Animation", 2022, None),
        BrandCompany("MGM Animation/Visual Arts", 2022, None),
        BrandCompany("MGM Family Entertainment", 2022, None),
        BrandCompany("MGM British Studios", 2022, None),
        BrandCompany("MGM Producción", 2022, None),
        # UA revival 2024
        BrandCompany("United Artists", 2024, None),
        BrandCompany("United Artists Pictures", 2024, None),
        BrandCompany("United Artists Film Corporation", 2024, None),
        BrandCompany("United Artists Europa", 2024, None),
    ))

    # APPLE_STUDIOS — producer-intent brand (platform intent handled by
    # watch_providers path).
    APPLE_STUDIOS = ("apple-studios", 31, "Apple Studios", (
        BrandCompany("Apple Original Films", None, None),
        BrandCompany("Apple Studios", None, None),
    ))


# ---------------------------------------------------------------------------
# Reverse index: surface string → [(brand, start_year, end_year), ...]
#
# Sorted alphabetically by brand.name (ascending), then by start_year
# (ascending, None as -inf), then by end_year (ascending, None as +inf).
# This matches the order produced by the EXPECTED_DATE_MEMBERSHIPS fixture
# in tests, and gives callers a deterministic iteration sequence.
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
    matches the user's ingestion-time directive: a movie with an unknown
    release year cannot be confidently bound to a time-gated brand.
    """
    if release_year is None:
        return start is None and end is None
    if start is not None and release_year < start:
        return False
    if end is not None and release_year > end:
        return False
    return True
