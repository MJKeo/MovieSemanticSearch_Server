# IMDB Keyword Vocabulary Audit

Audit of `overall_keywords` and `plot_keywords` from IMDB scraped data,
conducted against 109,238 qualifying movies (status `imdb_quality_passed`
or later in tracker.db).

---

## Key Finding: overall_keywords Is a Curated Genre Taxonomy

IMDB's `overall_keywords` is NOT a free-form community tagging system.
It is a compact, curated **genre/sub-genre taxonomy of exactly 225 terms**
with near-perfect coverage:

- 100.0% of qualifying movies have at least one `overall_keyword`
- Median 3 keywords per movie, mean 3.4, max 23
- Top 200 terms cover 99.8% of all keyword assignments
- Only 5 singletons in the entire vocabulary (2.2%)
- Zero overlap with `plot_keywords` (Jaccard = 0.000) -- completely
  disjoint vocabularies

This means **static mapping is trivially feasible** -- the entire
vocabulary can be hardcoded. No LLM translation or hybrid approach is
needed for `overall_keywords`.

## plot_keywords: Already Absorbed by Metadata Generation

`plot_keywords` is the free-form community tagging system (114,547
distinct terms, 53% appear exactly once, massive long tail). These raw
tags are already consumed by the metadata generation pipeline and
distilled into structured LLM metadata across the vector spaces. They do
not need a separate search path via `keyword_ids`.

---

## Vocabulary Structure

The 225 `overall_keywords` terms organize into these categories:

### Core Genres (~20 terms)

Standard IMDB genres that overlap with TMDB's genre system. These are
high-frequency tags present on most movies:

| Keyword | Movies | % |
|---------|-------:|--:|
| Drama | 56,781 | 52.0% |
| Comedy | 33,046 | 30.3% |
| Thriller | 23,565 | 21.6% |
| Romance | 20,156 | 18.5% |
| Action | 16,832 | 15.4% |
| Horror | 15,442 | 14.1% |
| Crime | 15,329 | 14.0% |
| Adventure | 10,570 | 9.7% |
| Mystery | 10,048 | 9.2% |
| Family | 9,094 | 8.3% |
| Documentary | 7,904 | 7.2% |
| Sci-Fi | 7,670 | 7.0% |
| Fantasy | 7,658 | 7.0% |
| Short | 5,642 | 5.2% |
| Biography | 4,571 | 4.2% |
| History | 4,196 | 3.8% |
| War | 4,168 | 3.8% |
| Music | 3,703 | 3.4% |
| Western | 3,696 | 3.4% |
| Musical | 3,318 | 3.0% |
| Sport | 2,407 | 2.2% |

Core genres function as deal-breakers when users specify genre
constraints ("show me horror movies"), but these are already well-served
by the existing TMDB genre IDs on `movie_card`. The incremental value
of `overall_keywords` core genres is minor.

### Sub-Genre Tags (~100 terms)

This is where `overall_keywords` provides the most deal-breaker value.
TMDB's ~20 genres cannot distinguish "folk horror" from "slasher horror"
or "spaghetti western" from "contemporary western." These tags can.

Organized by parent genre below with full deal-breaker concept mappings.

### Language/Nationality Tags (~30 terms)

Cinema tradition identifiers (French, Hindi, Korean, etc.). These are
useful as deal-breaker signals alongside the structured
`country_of_origin_ids` and `audio_language_ids` fields. The
overall_keyword language tag captures "cinema tradition" rather than
just production country -- a "French" tagged movie is part of the French
cinema tradition, which is what users typically mean when they search for
"French movies."

### Format Tags (~7 terms)

Stand-Up (123), Concert (85), Docudrama (703), News (172), plus TV
format noise (Sitcom 11, Reality TV 11, Talk Show 3, Game Show 1, etc.).
The TV formats are effectively noise for a movie search engine but
harmless to include.

---

## Deal-Breaker Concept Mappings

Each concept below maps user search intent to one or more
`overall_keywords` tags. The "user might say" column shows natural
language queries that should trigger keyword-based retrieval for this
concept.

### Production Medium

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Animation (any) | Animation, Hand-Drawn Animation, Computer Animation, Adult Animation, Stop Motion Animation, Anime, Mecha, Shonen, Seinen, Shojo, Isekai, Iyashikei, Slice of Life, Josei, Holiday Animation | 6,010 | "animated movies", "cartoons" |
| Hand-Drawn Animation | Hand-Drawn Animation | 1,591 | "hand-drawn animated films" |
| CGI Animation | Computer Animation | 792 | "CGI movies", "computer animated" |
| Stop Motion | Stop Motion Animation | 222 | "stop motion films", "claymation" |
| Adult Animation | Adult Animation | 751 | "adult animated movies" |
| Anime (any) | Anime, Mecha, Shonen, Seinen, Shojo, Isekai, Iyashikei, Slice of Life, Josei | 930 | "anime movies", "anime films" |

Note: Animation sub-type tags are subsets of the parent `Animation` tag
-- all 6,010 Animation movies have the `Animation` keyword, and
sub-types provide additional specificity.

### Holiday

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Holiday (any) | Holiday, Holiday Romance, Holiday Comedy, Holiday Family, Holiday Animation | 804 | "holiday movies", "christmas movies" |
| Holiday Romance | Holiday Romance | 402 | "holiday romance", "christmas romance" |
| Holiday Comedy | Holiday Comedy | 190 | "holiday comedies" |
| Holiday Family | Holiday Family | 139 | "family holiday movies" |

Coverage analysis: 83.8% of Holiday-tagged movies have explicit
"christmas" in their `plot_keywords`, confirming the tag is
overwhelmingly christmas-oriented. The remaining 16.2% includes:
- Christmas movies lacking the exact plot keyword (Home Alone 3, etc.)
- Holiday viewing traditions (Harry Potter, Groundhog Day, Die Hard)
- Other holidays: Thanksgiving, Hanukkah, New Year's, Easter

The Holiday overall_keyword is a curated editorial signal that captures
human judgment about "this is a holiday movie" -- strictly more useful
than checking for "christmas" in plot_keywords.

### Horror Sub-Genres

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Supernatural Horror | Supernatural Horror | 1,400 | "supernatural horror", "ghost movies" |
| Slasher | Slasher Horror | 1,144 | "slasher films", "slashers" |
| B-Horror | B-Horror | 983 | "B-movie horror", "cheesy horror" |
| Folk Horror | Folk Horror | 969 | "folk horror" |
| Body Horror | Body Horror | 907 | "body horror", "Cronenberg-style" |
| Psychological Horror | Psychological Horror | 763 | "psychological horror" |
| Monster Horror | Monster Horror | 475 | "monster movies" |
| Splatter/Gore | Splatter Horror | 393 | "splatter films", "gory horror" |
| Teen Horror | Teen Horror | 358 | "teen horror", "teen scream" |
| Zombie | Zombie Horror | 333 | "zombie movies" |
| Found Footage | Found Footage Horror | 309 | "found footage horror" |
| Vampire | Vampire Horror | 239 | "vampire movies" |
| Kaiju | Kaiju | 170 | "kaiju movies", "giant monster films" |
| Werewolf | Werewolf Horror | 134 | "werewolf movies" |
| Giallo | Giallo | 117 | "giallo", "Italian horror" |
| Witch Horror | Witch Horror | 67 | "witch movies" |

### Comedy Sub-Genres

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Dark Comedy | Dark Comedy | 3,447 | "dark comedies", "black comedy" |
| Satire | Satire | 1,854 | "satirical films", "satire" |
| Slapstick | Slapstick | 1,820 | "slapstick comedy" |
| Parody | Parody | 1,292 | "parody movies", "spoof films" |
| Romantic Comedy | Romantic Comedy | 1,227 | "rom-coms" |
| Farce | Farce | 626 | "farce", "farcical comedy" |
| Buddy Comedy | Buddy Comedy | 612 | "buddy comedies" |
| Quirky/Indie Comedy | Quirky Comedy | 553 | "quirky comedies", "indie comedy" |
| Raunchy Comedy | Raunchy Comedy | 477 | "raunchy comedy", "gross-out comedy" |
| Teen Comedy | Teen Comedy | 431 | "teen comedies" |
| Screwball Comedy | Screwball Comedy | 372 | "screwball comedy" |
| High-Concept Comedy | High-Concept Comedy | 261 | "high-concept comedy" |
| Buddy Cop | Buddy Cop | 106 | "buddy cop movies" |
| Stoner Comedy | Stoner Comedy | 90 | "stoner comedies" |
| Mockumentary | Mockumentary | 80 | "mockumentaries" |
| Sketch Comedy | Sketch Comedy | 77 | "sketch comedy" |
| Body Swap Comedy | Body Swap Comedy | 69 | "body swap movies" |

### Thriller Sub-Genres

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Psychological Thriller | Psychological Thriller | 2,312 | "psychological thrillers" |
| Conspiracy Thriller | Conspiracy Thriller | 384 | "conspiracy thrillers" |
| Erotic Thriller | Erotic Thriller | 363 | "erotic thrillers" |
| Political Thriller | Political Thriller | 321 | "political thrillers" |
| Cyber Thriller | Cyber Thriller | 99 | "cyber thrillers", "hacker movies" |
| Legal Thriller | Legal Thriller | 83 | "legal thrillers" |

### Drama Sub-Genres

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Psychological Drama | Psychological Drama | 3,027 | "psychological dramas" |
| Period Drama | Period Drama, Costume Drama | 2,206 | "period dramas", "costume dramas" |
| Tragedy | Tragedy | 1,094 | "tragedies", "tragic films" |
| Teen Drama | Teen Drama | 462 | "teen dramas" |
| Political Drama | Political Drama | 459 | "political dramas" |
| Workplace Drama | Workplace Drama | 223 | "workplace dramas", "office dramas" |
| Legal Drama | Legal Drama | 222 | "legal dramas", "courtroom dramas" |
| Showbiz Drama | Showbiz Drama | 210 | "showbiz dramas", "Hollywood movies about Hollywood" |
| Cop Drama | Cop Drama | 187 | "cop dramas", "police dramas" |
| Prison Drama | Prison Drama | 180 | "prison movies", "prison dramas" |
| Medical Drama | Medical Drama | 117 | "medical dramas", "hospital movies" |
| Financial Drama | Financial Drama | 61 | "Wall Street movies", "financial dramas" |

### Crime Sub-Genres

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| True Crime | True Crime | 832 | "true crime" |
| Whodunnit | Whodunnit | 693 | "whodunnits", "murder mystery" |
| Heist | Heist, Caper | 538 | "heist movies", "caper films" |
| Gangster | Gangster | 383 | "gangster movies", "mob films" |
| Serial Killer | Serial Killer | 281 | "serial killer movies" |
| Police Procedural | Police Procedural | 155 | "police procedurals" |
| Drug Crime | Drug Crime | 115 | "drug movies", "narco films" |
| Cozy Mystery | Cozy Mystery | 90 | "cozy mysteries" |

### Western Sub-Genres

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Classical Western | Classical Western | 400 | "classic westerns" |
| Spaghetti Western | Spaghetti Western | 393 | "spaghetti westerns" |
| Contemporary/Neo-Western | Contemporary Western | 102 | "modern westerns", "neo-westerns" |

### Sci-Fi Sub-Genres

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Space Sci-Fi | Space Sci-Fi | 552 | "space movies", "sci-fi in space" |
| Dystopian Sci-Fi | Dystopian Sci-Fi | 371 | "dystopian movies" |
| Cyberpunk | Cyberpunk | 173 | "cyberpunk films" |
| Artificial Intelligence | Artificial Intelligence | 147 | "AI movies" |
| Steampunk | Steampunk | 67 | "steampunk movies" |

### Fantasy Sub-Genres

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Supernatural Fantasy | Supernatural Fantasy | 492 | "supernatural fantasy" |
| Dark Fantasy | Dark Fantasy | 488 | "dark fantasy" |
| Fairy Tale | Fairy Tale | 236 | "fairy tale movies" |
| Sword & Sorcery | Sword & Sorcery | 162 | "sword and sorcery" |

### Action Styles

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Martial Arts (any) | Martial Arts, Kung Fu, Gun Fu, Wuxia, Samurai | 991 | "martial arts movies" |
| One-Person Army | One-Person Army Action | 533 | "one-man army", "Rambo-style" |
| Gun Fu | Gun Fu | 146 | "gun fu", "John Wick-style" |
| Kung Fu | Kung Fu | 140 | "kung fu movies" |
| Samurai | Samurai | 92 | "samurai films" |
| Wuxia | Wuxia | 85 | "wuxia movies" |
| Car Action | Car Action | 137 | "car chase movies" |

### Romance Sub-Genres

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Feel-Good Romance | Feel-Good Romance | 454 | "feel-good romance", "light romance" |
| Tragic Romance | Tragic Romance | 375 | "tragic love stories" |
| Teen Romance | Teen Romance | 245 | "teen romance" |
| Steamy Romance | Steamy Romance | 240 | "steamy romance" |
| Dark Romance | Dark Romance | 233 | "dark romance" |

### Thematic Concepts

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Coming-of-Age | Coming-of-Age | 1,252 | "coming of age movies" |
| Film Noir | Film Noir | 926 | "film noir", "noir" |
| Superhero | Superhero | 839 | "superhero movies" |
| Epic (any) | Epic, Action Epic, Adventure Epic, Sci-Fi Epic, Fantasy Epic, Romantic Epic, War Epic, Western Epic, Historical Epic | 1,171 | "epic movies", "epic films" |
| Spy | Spy | 366 | "spy movies", "espionage films" |
| Survival | Survival | 319 | "survival movies" |
| Road Trip | Road Trip | 269 | "road trip movies" |
| Disaster | Disaster | 247 | "disaster movies" |
| Time Travel | Time Travel | 224 | "time travel movies" |

### Adventure Sub-Types

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Animal Adventure | Animal Adventure | 486 | "animal movies", "movies about animals" |
| Quest | Quest | 423 | "quest movies" |
| Jungle Adventure | Jungle Adventure | 331 | "jungle adventure" |
| Urban Adventure | Urban Adventure | 252 | "urban adventure" |
| Swashbuckler | Swashbuckler | 227 | "swashbuckler", "pirate movies" |
| Sea Adventure | Sea Adventure | 202 | "ocean movies", "sea adventure" |
| Alien Invasion | Alien Invasion | 374 | "alien invasion movies" |
| Globetrotting Adventure | Globetrotting Adventure | 159 | "globetrotting adventure" |
| Dinosaur Adventure | Dinosaur Adventure | 145 | "dinosaur movies" |
| Desert Adventure | Desert Adventure | 136 | "desert movies" |
| Mountain Adventure | Mountain Adventure | 99 | "mountain movies" |
| Teen Adventure | Teen Adventure | 161 | "teen adventure" |

### Documentary Sub-Types

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Music Documentary | Music Documentary | 478 | "music documentaries" |
| Crime Documentary | Crime Documentary | 243 | "true crime docs" |
| Sports Documentary | Sports Documentary | 182 | "sports documentaries" |
| Nature Documentary | Nature Documentary | 100 | "nature docs" |
| Faith & Spirituality Doc | Faith & Spirituality Documentary | 86 | "religious documentaries" |
| Political Documentary | Political Documentary | 84 | "political documentaries" |
| History Documentary | History Documentary | 81 | "history docs" |
| Travel Documentary | Travel Documentary | 66 | "travel documentaries" |
| Military Documentary | Military Documentary | 63 | "military documentaries" |
| Science & Tech Documentary | Science & Technology Documentary | 52 | "science documentaries" |
| Food Documentary | Food Documentary | 50 | "food documentaries" |

### Sports

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Boxing | Boxing | 134 | "boxing movies" |
| Basketball | Basketball | 117 | "basketball movies" |
| Soccer | Soccer | 92 | "soccer movies", "football movies" (non-US) |
| Baseball | Baseball | 85 | "baseball movies" |
| Football | Football | 75 | "football movies" (US) |
| Motorsport | Motorsport | 69 | "racing movies" |
| Extreme Sport | Extreme Sport | 64 | "extreme sports movies" |
| Water Sport | Water Sport | 58 | "surfing movies" |

### Musical Sub-Types

| Concept | Keywords | Movies | User might say |
|---------|----------|-------:|----------------|
| Pop Musical | Pop Musical | 129 | "pop musicals" |
| Classic Musical | Classic Musical | 121 | "classic musicals", "old musicals" |
| Jukebox Musical | Jukebox Musical | 72 | "jukebox musicals" |
| Rock Musical | Rock Musical | 49 | "rock musicals" |

### Language/Nationality

30 cinema tradition tags. Full list with counts:

| Keyword | Movies | | Keyword | Movies |
|---------|-------:|-|---------|-------:|
| French | 4,440 | | Swedish | 521 |
| Hindi | 3,041 | | Turkish | 518 |
| Japanese | 2,877 | | Dutch | 397 |
| Spanish | 2,874 | | Arabic | 403 |
| Italian | 2,528 | | Danish | 353 |
| German | 1,801 | | Bengali | 293 |
| Mandarin | 1,378 | | Filipino | 282 |
| Tamil | 1,182 | | Norwegian | 251 |
| Korean | 1,061 | | Thai | 243 |
| Cantonese | 970 | | Kannada | 233 |
| Russian | 901 | | Persian | 219 |
| Malayalam | 878 | | Greek | 212 |
| Telugu | 857 | | Finnish | 178 |
| Portuguese | 569 | | Marathi | 158 |
| | | | Punjabi | 141 |
| | | | Urdu | 92 |

### Format/Other

| Concept | Keywords | Movies |
|---------|----------|-------:|
| Docudrama | Docudrama | 703 |
| Suspense Mystery | Suspense Mystery | 607 |
| B-Action | B-Action | 440 |
| Stand-Up Special | Stand-Up | 123 |
| Concert Film | Concert | 85 |
| Hard-boiled Detective | Hard-boiled Detective | 63 |
| Bumbling Detective | Bumbling Detective | 65 |
| Sword & Sandal | Sword & Sandal | 81 |

### Noise (TV formats, negligible counts)

Sitcom (11), Reality TV (11), Talk Show (3), Game Show (1),
Cooking Competition (1), Soap Opera (1), Business Reality TV (1),
Paranormal Reality TV (1). These leaked in from non-movie titles that
passed quality filtering. Harmless to include in `keyword_ids` but
effectively unused for search.

---

## Mapping Approach: Static Only

Given the findings:
1. Only 225 distinct terms -- fully enumerable
2. Near-zero long tail (5 singletons)
3. 100% movie coverage
4. Zero overlap with `plot_keywords`

**Recommendation: Pure static mapping.** No LLM translation needed for
the `overall_keywords` to `keyword_ids` mapping (task #11 in
v2_data_needs.md).

For query understanding, the Phase 0 LLM needs to map user natural
language to relevant `overall_keywords` terms. Two approaches:

1. **Include the full 225-term vocabulary in the QU prompt** -- small
   enough to fit easily. The LLM selects matching terms from the
   provided list. This is the simplest approach and likely sufficient.

2. **Static synonym expansion table** -- map common user phrases to
   keywords (e.g., "rom-com" -> Romantic Comedy, "noir" -> Film Noir,
   "Bollywood" -> Hindi). Used as a preprocessing step before or
   alongside the LLM.

Option 1 is recommended as the starting point. The vocabulary is small
enough that providing it as context to the QU LLM is trivial, and the
LLM can handle synonym resolution (rom-com, noir, etc.) without a
separate lookup table.

---

## Implications for V2 Design

### keyword_ids field
Proceed as originally scoped: map `overall_keywords` only (not
`plot_keywords`) to `lex.lexical_dictionary` string IDs and store as
`movie_card.keyword_ids` with a GIN index.

### Deal-breaker value
The primary value is **sub-genre precision across the entire genre
space** -- exactly the kind of deterministic signal that vector search is
weakest at. When a user asks for "folk horror" or "spaghetti westerns",
keyword matching provides a binary, high-confidence filter that vector
similarity can only approximate.

### Integration with Phase 0/1
Keywords function as a boost signal in deal-breaker retrieval (not a
hard pre-filter). Movies matching the keyword get automatic pass on the
deal-breaker threshold; movies without the keyword can still enter via
vector similarity. This preserves recall while boosting precision for
keyword-matchable concepts.

### Language tags alongside structured fields
The language/nationality tags in `overall_keywords` complement rather
than replace `country_of_origin_ids` and `audio_language_ids`. A query
for "Korean cinema" should check both the `Korean` keyword tag AND the
country/language structured fields. The keyword tag captures editorial
judgment about cinema tradition membership, which doesn't always align
with production country alone.
