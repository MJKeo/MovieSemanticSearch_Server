from __future__ import annotations

from typing import TypeAlias

BrandMembership: TypeAlias = tuple[str, int | None, int | None]


def _extend(
    mapping: dict[str, list[BrandMembership]],
    membership: BrandMembership,
    strings: list[str],
) -> None:
    for s in strings:
        mapping.setdefault(s, []).append(membership)


_EXPECTED: dict[str, list[BrandMembership]] = {}

_extend(_EXPECTED, ("AMAZON_MGM", 2022, None), [
    "MGM Animation/Visual Arts",
    "MGM British Studios",
    "MGM Family Entertainment",
    "MGM Producción",
    "Metro-Goldwyn-Mayer (MGM)",
    "Metro-Goldwyn-Mayer (MGM) Studios",
    "Metro-Goldwyn-Mayer Animation",
    "Metro-Goldwyn-Mayer Studios",
])

_extend(_EXPECTED, ("AMAZON_MGM", 2024, None), [
    "United Artists",
    "United Artists Europa",
    "United Artists Film Corporation",
    "United Artists Pictures",
])

_extend(_EXPECTED, ("BLUMHOUSE", 2008, None), [
    "Blumhouse Television",
])

_extend(_EXPECTED, ("BLUMHOUSE", 2024, None), [
    "Atomic Monster",
])

_extend(_EXPECTED, ("COLUMBIA", 1924, 1968), [
    "Columbia Pictures Corporation",
])

_extend(_EXPECTED, ("COLUMBIA", 1924, None), [
    "Columbia British Productions",
    "Columbia Films",
    "Columbia Pictures",
    "Columbia Productions",
])

_extend(_EXPECTED, ("DC", 2009, 2016), [
    "DC Entertainment",
])

_extend(_EXPECTED, ("DC", 2016, 2022), [
    "DC Films",
])

_extend(_EXPECTED, ("DC", 2022, None), [
    "DC Studios",
])

_extend(_EXPECTED, ("DC", None, 2009), [
    "DC Comics",
])

_extend(_EXPECTED, ("DISNEY", 1923, None), [
    "Walt Disney Animation Australia",
    "Walt Disney Animation Canada",
    "Walt Disney Animation France S.A.",
    "Walt Disney Animation Japan",
    "Walt Disney Animation Studios",
    "Walt Disney Feature Animation",
    "Walt Disney Feature Animation Florida",
    "Walt Disney Feature Animation Paris",
])

_extend(_EXPECTED, ("DISNEY", 1929, 1986), [
    "Walt Disney Productions",
])

_extend(_EXPECTED, ("DISNEY", 1983, None), [
    "Walt Disney British Films",
    "Walt Disney Home Entertainment",
    "Walt Disney Home Video",
    "Walt Disney Pictures",
    "Walt Disney Pictures / Sony Pictures",
    "Walt Disney Studios",
    "Walt Disney Studios Home Entertainment",
    "Walt Disney Studios Motion Pictures",
])

_extend(_EXPECTED, ("DISNEY", 1984, 2018), [
    "Touchstone",
    "Touchstone Films",
    "Touchstone Pictures",
    "Touchstone Pictures México",
])

_extend(_EXPECTED, ("DISNEY", 1988, 2018), [
    "Disneytoon Studios",
])

_extend(_EXPECTED, ("DISNEY", 1989, 2007), [
    "Hollywood Pictures",
    "Hollywood Pictures Corporation (I)",
    "Hollywood Pictures Corporation (II)",
    "Hollywood Pictures Home Video",
])

_extend(_EXPECTED, ("DISNEY", 1993, 2005), [
    "Dimension Films",
    "Dimension Films (II)",
])

_extend(_EXPECTED, ("DISNEY", 1993, 2010), [
    "Miramax",
    "Miramax Family Films",
    "Miramax Home Entertainment",
    "Miramax International",
])

_extend(_EXPECTED, ("DISNEY", 2006, None), [
    "Pixar",
    "Pixar Animation Studios",
])

_extend(_EXPECTED, ("DISNEY", 2009, None), [
    "Marvel Studios",
])

_extend(_EXPECTED, ("DISNEY", 2012, None), [
    "Lucasfilm",
    "Lucasfilm Animation",
])

_extend(_EXPECTED, ("DISNEY", 2019, 2020), [
    "20th Century Fox",
    "20th Century Pictures",
    "Fox Searchlight Pictures",
    "Twentieth Century Animation",
    "Twentieth Century Fox",
    "Twentieth Century Fox Animation",
    "Twentieth Century Productions",
    "Twentieth Century-Fox Productions",
    "Twentieth Century-Fox Studios, Hollywood",
])

_extend(_EXPECTED, ("DISNEY", 2019, 2021), [
    "Blue Sky Studios",
])

_extend(_EXPECTED, ("DISNEY", 2019, None), [
    "20th Century Fox Argentina",
    "20th Century Fox Home Entertainment",
    "20th Century Fox Korea",
    "20th Century Fox Post Production Services",
    "20th Century Studios",
    "Searchlight Pictures",
])

_extend(_EXPECTED, ("DREAMWORKS_ANIMATION", 1994, 2015), [
    "Pacific Data Images (PDI)",
])

_extend(_EXPECTED, ("DREAMWORKS_ANIMATION", 2004, None), [
    "DreamWorks Animation",
])

_extend(_EXPECTED, ("FOCUS_FEATURES", 1991, 2002), [
    "Good Machine",
    "Good Machine Films",
])

_extend(_EXPECTED, ("FOCUS_FEATURES", 1999, 2002), [
    "USA Films",
])

_extend(_EXPECTED, ("FOCUS_FEATURES", 2002, None), [
    "Focus Features",
    "Focus Features Africa First Program",
    "Focus Features International (FFI)",
])

_extend(_EXPECTED, ("FOCUS_FEATURES", 2010, 2013), [
    "Focus World",
])

_extend(_EXPECTED, ("ILLUMINATION", 2007, 2018), [
    "Illumination Entertainment",
])

_extend(_EXPECTED, ("ILLUMINATION", 2011, None), [
    "Illumination Studios Paris",
    "Mac Guff Ligne",
])

_extend(_EXPECTED, ("LIONSGATE", 2003, None), [
    "Artisan Entertainment",
])

_extend(_EXPECTED, ("LIONSGATE", 2007, None), [
    "Mandate Pictures",
])

_extend(_EXPECTED, ("LIONSGATE", 2012, None), [
    "Summit Entertainment",
    "Summit Premiere",
])

_extend(_EXPECTED, ("LUCASFILM", 1971, None), [
    "Lucasfilm",
])

_extend(_EXPECTED, ("MARVEL_STUDIOS", 1993, 1996), [
    "Marvel Films",
])

_extend(_EXPECTED, ("MARVEL_STUDIOS", 1996, None), [
    "Marvel Studios",
])

_extend(_EXPECTED, ("MGM", 1924, None), [
    "Metro-Goldwyn-Mayer (MGM)",
])

_extend(_EXPECTED, ("MGM", 1936, 1970), [
    "MGM British Studios",
    "Metro-Goldwyn-Mayer British Studios",
])

_extend(_EXPECTED, ("MGM", 1937, 1957), [
    "MGM Animation/Visual Arts",
    "Metro-Goldwyn-Mayer Animation",
    "Metro-Goldwyn-Mayer Cartoon Studios",
])

_extend(_EXPECTED, ("MGM", 1986, None), [
    "MGM Family Entertainment",
    "MGM Producción",
    "Metro-Goldwyn-Mayer (MGM) Studios",
    "Metro-Goldwyn-Mayer Studios",
])

_extend(_EXPECTED, ("MGM", 1990, 1992), [
    "MGM-Pathé Communications Co.",
])

_extend(_EXPECTED, ("MIRAMAX", 1979, None), [
    "Miramax",
])

_extend(_EXPECTED, ("MIRAMAX", 1992, 2005), [
    "Dimension Films",
])

_extend(_EXPECTED, ("NEW_LINE_CINEMA", 1967, None), [
    "New Line Cinema",
    "New Line Film",
    "New Line Film Productions",
    "New Line Productions",
])

_extend(_EXPECTED, ("NEW_LINE_CINEMA", 1990, 2005), [
    "Fine Line Features",
])

_extend(_EXPECTED, ("PARAMOUNT", 1995, None), [
    "Nickelodeon Animation Studios",
    "Nickelodeon Films",
    "Nickelodeon Movies",
])

_extend(_EXPECTED, ("PARAMOUNT", 1996, None), [
    "MTV Entertainment Studios",
    "MTV Films",
    "MTV Films Europe",
])

_extend(_EXPECTED, ("PARAMOUNT", 2006, 2013), [
    "Paramount Classics",
    "Paramount Vantage",
])

_extend(_EXPECTED, ("PARAMOUNT", 2011, None), [
    "Paramount Animation",
    "Paramount Animation Studios",
])

_extend(_EXPECTED, ("PARAMOUNT", 2017, None), [
    "Paramount Players",
])

_extend(_EXPECTED, ("PARAMOUNT", 2023, None), [
    "Republic Pictures",
    "Republic Pictures (III)",
])

_extend(_EXPECTED, ("PIXAR", 1986, None), [
    "Pixar",
    "Pixar Animation Studios",
])

_extend(_EXPECTED, ("SEARCHLIGHT", 1994, 2020), [
    "Fox Searchlight Pictures",
])

_extend(_EXPECTED, ("SEARCHLIGHT", 2020, None), [
    "Searchlight Pictures",
])

_extend(_EXPECTED, ("SONY", 1989, None), [
    "Columbia",
    "Columbia British Productions",
    "Columbia Films",
    "Columbia Films Productions",
    "Columbia Pictures",
    "Columbia Pictures Corporation",
    "Columbia Pictures Entertainment",
    "Columbia Pictures Film Production Asia",
    "Columbia Pictures Industries",
    "Columbia Pictures Producciones Mexico",
    "Columbia Pictures do Brasil",
    "Columbia Pictures of Argentina",
    "Columbia Pictures of Brasil",
    "Columbia Productions",
    "Columbia Release",
    "Deutsche Columbia Pictures Film Produktion",
    "Tri Star",
    "Tri Star Productions",
    "Tri-Star Pictures",
    "TriStar Pictures",
    "TriStar Productions",
    "Triumph Films",
    "Triumph Films (II)",
])

_extend(_EXPECTED, ("SONY", 1992, None), [
    "Sony Pictures Classics",
])

_extend(_EXPECTED, ("SONY", 1999, None), [
    "Screen Gems",
])

_extend(_EXPECTED, ("SONY", 2002, None), [
    "Sony Pictures Animation",
])

_extend(_EXPECTED, ("SONY", 2007, None), [
    "Stage 6 Films",
    "Stage 6 Productions",
])

_extend(_EXPECTED, ("SONY_PICTURES_ANIMATION", 2002, None), [
    "Sony Pictures Animation",
])

_extend(_EXPECTED, ("TOUCHSTONE", 1984, 1986), [
    "Touchstone Films",
])

_extend(_EXPECTED, ("TOUCHSTONE", 1986, 2018), [
    "Touchstone Pictures",
])

_extend(_EXPECTED, ("TRISTAR", 1982, 1991), [
    "Tri-Star Pictures",
])

_extend(_EXPECTED, ("TRISTAR", 1982, None), [
    "Tri Star",
    "TriStar Pictures",
])

_extend(_EXPECTED, ("TRISTAR", 2015, None), [
    "TriStar Productions",
])

_extend(_EXPECTED, ("TWENTIETH_CENTURY", 1935, 2020), [
    "20th Century Foss",
    "20th Century Fox",
    "20th Century Pictures",
    "Twentieth Century Fox",
    "Twentieth Century Fox Animation",
    "Twentieth Century Productions",
    "Twentieth Century-Fox Productions",
    "Twentieth Century-Fox Studios, Hollywood",
])

_extend(_EXPECTED, ("TWENTIETH_CENTURY", 1994, 2020), [
    "Fox 2000 Pictures",
])

_extend(_EXPECTED, ("TWENTIETH_CENTURY", 2020, None), [
    "20th Century Studios",
    "Twentieth Century Animation",
])

_extend(_EXPECTED, ("TWENTIETH_CENTURY", None, 1934), [
    "Fox Film Company",
    "Fox Film Corporation",
    "Fox Films",
])

_extend(_EXPECTED, ("UNITED_ARTISTS", 1919, 1981), [
    "United Artists",
])

_extend(_EXPECTED, ("UNITED_ARTISTS", 1981, 2018), [
    "United Artists Europa",
    "United Artists Film Corporation",
    "United Artists Pictures",
])

_extend(_EXPECTED, ("UNITED_ARTISTS", 2024, None), [
    "United Artists",
])

_extend(_EXPECTED, ("UNIVERSAL", 1992, 1999), [
    "Gramercy Pictures (I)",
    "Gramercy Pictures (II)",
])

_extend(_EXPECTED, ("UNIVERSAL", 1999, None), [
    "WT2 Productions",
    "Working Title Films",
])

_extend(_EXPECTED, ("UNIVERSAL", 2002, None), [
    "Focus Features",
    "Focus Features Africa First Program",
    "Focus Features International (FFI)",
    "Focus World",
    "Good Machine",
    "Good Machine Films",
    "USA Films",
])

_extend(_EXPECTED, ("UNIVERSAL", 2007, None), [
    "Illumination Entertainment",
])

_extend(_EXPECTED, ("UNIVERSAL", 2011, None), [
    "Mac Guff Ligne",
])

_extend(_EXPECTED, ("UNIVERSAL", 2016, None), [
    "DreamWorks Animation",
    "Pacific Data Images (PDI)",
])

_extend(_EXPECTED, ("WALT_DISNEY_ANIMATION", 1929, 1986), [
    "Walt Disney Productions",
])

_extend(_EXPECTED, ("WALT_DISNEY_ANIMATION", 1986, 2007), [
    "Walt Disney Feature Animation",
    "Walt Disney Feature Animation Florida",
    "Walt Disney Feature Animation Paris",
])

_extend(_EXPECTED, ("WALT_DISNEY_ANIMATION", 2007, None), [
    "Walt Disney Animation Australia",
    "Walt Disney Animation Canada",
    "Walt Disney Animation France S.A.",
    "Walt Disney Animation Japan",
    "Walt Disney Animation Studios",
])

_extend(_EXPECTED, ("WARNER_BROS", 1923, None), [
    "Warner Bros Entertainment",
    "Warner Bros.",
    "Warner Bros. Entertainment",
    "Warner Bros. First National",
    "Warner Bros. Pictures",
    "Warner Bros. Productions",
    "Warner Bros.-First National Pictures",
    "Warner Bros./Seven Arts",
    "Warner Brothers Entertainment",
    "Warner Brothers Pictures",
    "Warner Brothers-First National Productions",
])

_extend(_EXPECTED, ("WARNER_BROS", 1980, None), [
    "Warner Bros. Animation",
    "Warner Bros. Cartoon Studios",
    "Warner Bros. Feature Animation",
    "Warner Bros. New York Animation",
    "Warner Bros. Pictures Animation",
    "Warner Brothers/Seven Arts Animation",
    "Warner Classic Animation",
])

_extend(_EXPECTED, ("WARNER_BROS", 1989, 1999), [
    "HBO Pictures",
])

_extend(_EXPECTED, ("WARNER_BROS", 1996, 1998), [
    "Ted Turner Pictures",
    "Turner Pictures (I)",
    "Turner Pictures (III)",
    "Turner Pictures Worldwide",
])

_extend(_EXPECTED, ("WARNER_BROS", 1996, None), [
    "Castle Rock Entertainment",
    "Fine Line Features",
    "New Line Cinema",
    "New Line Film",
    "New Line Film Productions",
    "New Line Productions",
])

_extend(_EXPECTED, ("WARNER_BROS", 1999, None), [
    "HBO Documentary Films",
    "HBO Films",
    "HBO Max",
    "HBO Premiere Films",
    "Home Box Office (HBO)",
])

_extend(_EXPECTED, ("WARNER_BROS", 2009, 2022), [
    "DC Entertainment",
])

_extend(_EXPECTED, ("WARNER_BROS", 2016, 2022), [
    "DC Films",
])

_extend(_EXPECTED, ("WARNER_BROS", 2022, None), [
    "DC Studios",
])

_extend(_EXPECTED, ("WARNER_BROS", None, 2009), [
    "DC Comics",
])

EXPECTED_DATE_MEMBERSHIPS: dict[str, tuple[BrandMembership, ...]] = {
    surface: tuple(rows)
    for surface, rows in sorted(_EXPECTED.items())
}
