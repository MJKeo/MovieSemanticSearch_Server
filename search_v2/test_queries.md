# Step 2 Test Queries

A spread of queries written the way people actually type into a
phone keyboard or TV remote — casual, lowercase, with typos,
abbreviations, and run-ons. Each query is followed by two thought
processes:

1. **Mental model** — how I'd actually think through the request
   if a friend asked, free of any database or pipeline shackles.
2. **With backing data** — how I'd approach it given access to
   what we have (lexical entities, structured movie metadata, and
   eight vector spaces) but unconstrained by any specific search
   architecture.

The point is to surface the full thought process, not a final
answer — so we can see what the pipeline really needs to support.

**Backing data layout, for reference:**
- Lexical entities: actors, directors, franchises, characters
- Movie metadata: title, release date, genre, runtime, MPAA
  rating, streaming providers, audience/critic scores, awards,
  budget, box office
- Eight vector spaces per movie:
  `dense_anchor` (core thematic summary),
  `plot_events` (what happens / characters),
  `plot_analysis` (themes, arcs, concepts),
  `narrative_techniques` (storytelling style),
  `viewer_experience` (emotional tone, pacing),
  `watch_context` (when / how / with-whom to watch),
  `production` (technical achievements, source-of-inspiration
  style),
  `reception` (critical reception, awards, audience reaction)

---

## 1. Single bare attribute

**Query:** `scary`

**Mental model:** "Scary" is bare, but the intent is clearly
horror — or at least films designed to actually frighten. The
ambiguity is *flavor* and *intensity*: psychological dread
(Hereditary, The Babadook), supernatural shock (Insidious,
Sinister), classic monster (Alien, The Thing), or slasher
(Halloween, Scream). Without more context I'd return a spread of
genuinely-frightening, well-regarded horror across those flavors,
biased toward films most people agree actually deliver fear
rather than seasonal-spooky picks.

**With backing data:** Genre = horror as a soft prior, then lean
hard on `viewer_experience` for "scary, frightening, terrifying"
and `reception` for films audiences/critics agree deliver fear.
Lexical entities do nothing here. I wouldn't subgenre-filter; let
the vector spread find the variety. Quality reranking matters
because "scary" without specifics implies *reliably* scary, not
just any horror in the catalog.

---

## 2. Director name with typo + runtime polarity

**Query:** `quentin terantino movies that arent too long`

**Mental model:** Tarantino films, but the user is gun-shy about
runtime — Hateful Eight is 3+ hours, Once Upon a Time in
Hollywood is 2:40, Kill Bill Vol 1+2 together is huge. So I'd
recommend his shorter ones: Reservoir Dogs (99 min), Pulp Fiction
(154 min — borderline), Jackie Brown (154 min — borderline),
Death Proof (113 min). Reservoir Dogs is the easy win.

**With backing data:** Lexical lookup on "terantino" → fuzzy
match → Tarantino as director. Filter movies where he's credited
as director. Apply a soft runtime cap (probably ~135 min for "not
too long"). Sort by `reception` within the filter. Vectors don't
add much for a director-name query, but `reception` could break
ties between his shorter films.

---

## 3. Texting shorthand and abbreviations

**Query:** `luv stories w sad endings 4 a rainy night`

**Mental model:** Romance, but specifically the tragic kind, plus
a melancholic-mood watching context. Atonement (sad ending +
atmospheric), Eternal Sunshine, Blue Valentine, The Notebook,
Brokeback Mountain, Marriage Story, Past Lives. Atonement and
Eternal Sunshine fit the rainy-night register specifically — the
sad ending is part of the appeal.

**With backing data:** Genre = romance/drama soft prior.
`plot_events` for "tragic ending, lovers separated" — watch out
for false positives that have sad scenes but happy endings.
`viewer_experience` for "melancholy, bittersweet, contemplative"
catches the rainy-night mood. `watch_context` reinforces it.
Texting shorthand needs to be normalized before any lexical
lookup but kept verbatim for the LLM's reasoning.

---

## 4. Long run-on voice query, polarity stacking

**Query:** `i wanna watch something with my mom shes 65 and likes cozy mysteries nothing too dark or scary please`

**Mental model:** Co-viewing with a 65-year-old mother who likes
cozy mysteries — the explicit positive ask. So Knives Out / Glass
Onion (perfect fit), See How They Run, the Murder on the Orient
Express / Death on the Nile reboots, Only Murders in the Building
(TV but the same vibe). Has to feel safe — no graphic violence,
no real horror — and the mother's preferences anchor the choice
more than the user's.

**With backing data:** Genre = mystery with comedy adjacency.
`watch_context` for "co-viewing with parent, gentle, low-stakes."
`viewer_experience` filtered AGAINST "dark, scary, gruesome."
MPAA PG-13 or below, possibly. Reception bias high since this is
a recommendation, not exploration. The 65 + mom specifics aren't
hard filters but they sharpen the watch_context lookup.

---

## 5. Two simultaneous role markers on different people

**Query:** `directed by david lynch starring kyle maclachlan`

**Mental model:** Lynch + MacLachlan = Blue Velvet, Dune (1984),
Twin Peaks: Fire Walk With Me. Tiny intersection, and all three
are well known. I'd lead with Blue Velvet.

**With backing data:** Two lexical lookups — director = Lynch,
cast contains MacLachlan. Hard intersection on credits. Sort by
reception. Vectors do nothing useful here; this is a structured
metadata query masquerading as natural language.

---

## 6. Meta-relation parody (CLARIFYING EVIDENCE rule)

**Query:** `parody of the godfather`

**Mental model:** This is hard — direct Godfather parodies are
rare. Mafia! (1998) is the most explicit one. The Freshman (1990)
plays with Brando's Don Corleone persona. Johnny Dangerously is
gangster-adjacent satire. Analyze This riffs on tropes but isn't
a Godfather parody. The honest answer is "the field is thin" and
to surface what exists.

**With backing data:** The Godfather as a *reference point*, not
the answer — the meta-relation "parody of" rules out the film
itself. `production` / `plot_analysis` for "mafia satire,
gangster parody." Genre = comedy. Maybe a structural compare-vs
endpoint that takes the Godfather's vector embeddings and looks
for comedies whose plot/style references it.

---

## 7. Plot description with no parametric anchor

**Query:** `the one where the guy keeps reliving the same day`

**Mental model:** Groundhog Day is the canonical answer. But the
trope has spread: Edge of Tomorrow, Happy Death Day (1+2), Palm
Springs, Source Code, Before I Fall, Map of Tiny Perfect Things.
The user might want the original specifically or the trope
broadly. I'd lead with Groundhog Day and offer the rest.

**With backing data:** Pure `plot_events` vector territory —
"time loop, reliving the same day, repeating day." No lexical or
metadata anchors. Don't try to identify a single title; return
ranked candidates by plot similarity and let the user pick.

---

## 8. Two parametric references each with comparison polarity

**Query:** `darker than fight club but funnier than seven`

**Mental model:** Fight Club is already dark (consumerism,
violence, mental illness), so "darker" pushes toward extreme —
American Psycho, Fincher's own work, Funny Games. Se7en is
humorless, so "funnier than" is a low bar — any black-comedy
element clears it. American Psycho is the precise hit (darker
than FC, blackly comic). In Bruges or The Lobster also work.

**With backing data:** Both films are reference points, not
targets. Pull each film's `viewer_experience` vector — Fight
Club's "darkness" coordinate, Se7en's "humor" coordinate. Find
films whose darkness > FC's and humor > Se7en's. This is a
two-axis comparison endpoint, not a similarity search.

---

## 9. Multi-dimension entity + negative chronological

**Query:** `wonder woman movies but not the new ones`

**Mental model:** "The new ones" excludes the Gal Gadot films
(2017, 2020). So... the 1974 TV movie? The 2009 animated film?
The honest answer is the field is small once you exclude the
recent live-action — I'd surface what exists and admit the
narrowness rather than overreach.

**With backing data:** Franchise = Wonder Woman. Date filter
excluding 2017+. The result set is genuinely tiny. The pipeline
needs to surface that smallness rather than backfill with
loosely-related Wonder Woman content.

---

## 10. Title-character exclusion by specific instance

**Query:** `joker but not the joaquin phoenix one`

**Mental model:** Joker the character, excluding the 2019/2024
Phillips/Phoenix films. So Heath Ledger in The Dark Knight (the
canonical answer), Jack Nicholson in Batman (1989), Cesar Romero
in the 1966 Batman, Jared Leto in Suicide Squad (widely panned).
The Dark Knight is the lead.

**With backing data:** Character = Joker as a soft anchor —
films featuring the Joker. Exclude where lead/title actor is
Joaquin Phoenix. The exclusion narrows the set; it does NOT add
a positive Phoenix-credit signal somewhere else.

---

## 11. Negation of major franchises/studios

**Query:** `superhero movie not from marvel or dc`

**Mental model:** The Incredibles, Unbreakable / Glass /
Split (the Eastrail trilogy), Hancock, Chronicle, Kick-Ass,
Wanted, Spawn, Hellboy (technically Dark Horse), The Boys (TV).
The Incredibles and Unbreakable are the standouts.

**With backing data:** Genre = superhero. Franchise NOT IN
{Marvel Cinematic Universe, DC Extended Universe, X-Men, Sony
Spider-Man, ...}. Need a curated franchise-exclusion list since
"marvel or dc" is informal — does it cover X-Men? Sony? The list
needs explicit definition.

---

## 12. Precise era + niche sub-genre

**Query:** `early 2000s neo noir`

**Mental model:** Memento (2000), Mulholland Drive (2001), Brick
(2005), Sin City (2005), Collateral (2004), A History of Violence
(2005). Brick especially captures the niche — neo-noir tropes in
a high-school setting.

**With backing data:** Date range ~2000–2005. Genre/subgenre =
neo-noir (or noir + post-1990 as a proxy). Pretty clean
two-attribute intersection. `plot_analysis` could refine the
"noir" feel if the genre tag is too thin.

---

## 13. Platform + mood-occasion bundle

**Query:** `whats good on netflix when im hungover`

**Mental model:** Hungover = low cognitive load, easy pacing,
comforting, undemanding. Plus Netflix US currently. Plus quality.
So feel-good comedies, low-stakes adventures, comfort rewatches —
About Time, Glass Onion, Murder Mystery, Always Sunny (TV),
Pixar comfort food. The Netflix filter is dynamic so the answer
shifts month-to-month.

**With backing data:** Streaming = Netflix as a hard filter.
`watch_context` for "hungover, low effort, easy, undemanding."
`viewer_experience` for "comforting, light." Reception high. The
bundle decomposes into watch_context + viewer_experience without
inventing genre constraints.

---

## 14. Apparent contradiction that actually stacks

**Query:** `slow paced action movie`

**Mental model:** Drive, The American (2010), Le Samouraï,
Sicario, No Country for Old Men, Heat (mid-paced), The Proposition.
Drive is the prototype — action film with deliberate, contemplative
pacing.

**With backing data:** Genre = action AND `viewer_experience` =
"slow burn, deliberate pacing, contemplative." These read as
contradictory at the surface but are independent dimensions —
the system has to keep both as live constraints rather than
treating one as a typo.

---

## 15. Holiday occasion + negation of stylistic positioning

**Query:** `christmas movie thats actually good not the hallmark kind`

**Mental model:** Die Hard, Home Alone, Elf, National Lampoon's
Christmas Vacation, The Holdovers (2023), Klaus, A Christmas
Story, Tokyo Godfathers. The user wants quality + Christmas
setting, explicitly excluding the cheesy formulaic-romance niche.
The Holdovers is a fresh strong answer.

**With backing data:** Theme = Christmas (likely a holiday tag or
plot_events match). Reception > strong threshold. "Not Hallmark
kind" is hard to encode directly — closest proxies are
reception-score floor + studio exclusion + a `production` vector
filter against "made-for-TV romance" stylings.

---

## 16. Pure parallel filters — three independent attributes

**Query:** `80s action arnold`

**Mental model:** Three independent filters with a clear
intersection. Arnold's 80s action canon: Terminator (1984),
Predator (1987), Commando (1985), Conan the Barbarian (1982),
The Running Man (1987), Red Heat (1988), Raw Deal (1986).
Predator and Terminator are the leads.

**With backing data:** Cast contains Schwarzenegger AND date in
1980–1989 AND genre = action. Three-way structured intersection,
sort by reception. No vector work needed. The thing to test is
that none of the three attributes gets folded into the others
("80s arnold action" parses identically) and none gets demoted to
secondary.

---

## 17. Pure parallel filters — actor + decade + genre

**Query:** `90s comedy jim carrey`

**Mental model:** Carrey's 90s comedy run is iconic — Ace
Ventura (1994), The Mask (1994), Dumb and Dumber (1994), Liar
Liar (1997), Cable Guy (1996), Truman Show (1998 — comedy/drama
hybrid). His 90s output is mostly comedy by default.

**With backing data:** Same shape as #16 — cast = Carrey, date =
1990s, genre = comedy. Three-way intersection. The only
interesting wrinkle is The Truman Show being genre-ambiguous
(comedy/drama) — the system needs to handle soft genre
membership rather than strict include/exclude.

---

## 18. Multi-anchor reference set

**Query:** `inception interstellar tenet`

**Mental model:** Three Christopher Nolan sci-fi films listed
without connective tissue — the user wants the *common DNA*, not
a movie that IS those three. Common threads: Nolan, cerebral
sci-fi, time/physics as plot mechanics, dense plotting. So:
more Nolan (Memento, The Prestige, Oppenheimer), or other
cerebral mind-benders (Arrival, Primer, Coherence, Annihilation,
Ex Machina, The Matrix). Arrival is a strong cross-recommendation.

**With backing data:** This is the anchor-pattern hallucination
case — there's no single anchor. Pull each film's vector profile
and find common themes via intersection / centroid in
`plot_analysis` and `narrative_techniques`. Director match for
more Nolan. Crucially: do NOT score candidates against a fused
"sum of three films" — that produces nonsense. Score against
shared dimensions only.

---

## 19. Use-case scoping — explicit watching context

**Query:** `date night movie`

**Mental model:** Crowd-pleasing romance or rom-com, or a film
both partners enjoy regardless of genre. La La Land, About Time,
Crazy Rich Asians, When Harry Met Sally, Notting Hill, 10 Things
I Hate About You. Or broadly enjoyable non-romance: Knives Out,
The Princess Bride.

**With backing data:** This is exactly what `watch_context` is
built for — "date night" is a primary axis of that vector space.
Soft genre prior toward romance/dramedy. Reception bias high
since the cost of a bad recommendation in a date-night context
is high.

---

## 20. Use-case scoping — audience constraint

**Query:** `something for kids`

**Mental model:** G/PG family-friendly. Pixar, Disney, Studio
Ghibli, DreamWorks. Toy Story, Inside Out, Spirited Away,
Paddington, How to Train Your Dragon, The Iron Giant. Animation
dominates but not exclusively (Paddington, The Princess Bride).

**With backing data:** MPAA = G or PG hard filter. Genre =
family/animation soft prior. `watch_context` for "with kids."
Largely a structured query with a mild vector reinforcement.

---

## 21. Pure tonal / mood — figurative

**Query:** `warm hug movie`

**Mental model:** Pure mood, no concrete attributes. Paddington 2
is the canonical answer — virtually everyone uses it as the
exemplar. About Time, Paterson, Studio Ghibli (Totoro especially),
The Secret Life of Walter Mitty, Amélie, Hunt for the Wilderpeople.
Films that feel emotionally cozy regardless of genre.

**With backing data:** Pure `viewer_experience` territory —
"warm, gentle, cozy, comforting, kind." NO genre filter (these
span genres) and no plot filter. Reception bias for quality. The
LLM has to recognize that figurative tonal language is its own
valid criterion, not a placeholder for something else.

---

## 22. Pure tonal — common figurative

**Query:** `feel good film`

**Mental model:** Forrest Gump, Sing Street, Little Miss
Sunshine, School of Rock, The Intouchables, Hunt for the
Wilderpeople, CODA, Begin Again. Broad emotional uplift, often
with a journey or growth arc.

**With backing data:** `viewer_experience` for "uplifting,
heartwarming, feel-good." Some overlap with `watch_context` for
mood-lifting. No genre filter — feel-good spans drama, comedy,
musical, family.

---

## 23. Negation-only — no positive ask

**Query:** `anything but a romcom`

**Mental model:** This is a degenerate query — pure exclusion
with no positive content. The user is telling me what they DON'T
want and effectively asking me to pick. I'd treat this as
"recommend something good" with a romcom exclusion, and probably
ask a clarifying question if the channel allowed.

**With backing data:** Filter NOT genre = romcom. With nothing
else, default to broadly-recommended high-reception films across
remaining genres — basically "popular non-romcom." The pipeline
needs to recognize that negation-only queries have no positive
signal and handle them gracefully (default browse + the
exclusion) rather than failing or returning chaos.

---

## 24. Negation-heavy — multiple exclusions, no positive

**Query:** `no horror no romance`

**Mental model:** Same problem as #23, two exclusions. User
wants anything else. With 6+ remaining genres I'd default to a
spread of well-regarded films across action, drama, comedy,
sci-fi, thriller, animation.

**With backing data:** Filter out horror AND romance. No
positive vector signal. Default surface = high-reception
remaining-genre spread. The pipeline must handle the *list* of
negations without splitting them into separate dropped fragments.

---

## 25. Mixed positive + negative carvers

**Query:** `non violent crime thriller`

**Mental model:** Crime thrillers without the violence —
Catch Me If You Can, The Sting, Rounders, American Hustle,
Now You See Me, Inside Man, Ocean's Eleven (heist subset). Heist
films naturally fit. Catch Me If You Can is the cleanest answer.

**With backing data:** Genre = crime AND thriller. NEGATIVE on
violence — `viewer_experience` filtered against "violent, brutal,
gore." This is the hardest part: vectors don't negate cleanly,
so this might require a violence-level metadata tag (MPAA proxy
or ingested tag) or a filtered subset of the vector space.

---

## 26. Counterfactual — show + setting transposition

**Query:** `breaking bad but in the 1800s`

**Mental model:** Counterfactual style transfer. Breaking Bad's
essence: ordinary man's moral descent, criminal empire, anti-hero
who becomes the villain. Plus 1800s setting. Best matches: There
Will Be Blood (oil empire / moral descent / period), The Power
of the Dog (period anti-hero), Hell or High Water (modern western
spirit), McCabe & Mrs. Miller. There Will Be Blood is the closest
mood-and-trajectory match.

**With backing data:** Two-axis decomposition — Breaking Bad's
plot_analysis embedding ("anti-hero rise, moral descent, criminal
empire") AND time period = 1800s metadata. Combine. NOT a
similarity search to BB — that returns crime dramas in the wrong
era. The pipeline has to recognize "but in the 1800s" as a
setting-swap operator on BB's thematic profile.

---

## 27. Counterfactual — show + premise transposition

**Query:** `like succession but with pirates`

**Mental model:** Succession's essence: dysfunctional dynasty
fighting over inheritance, dark satire, biting dialogue. Plus
pirate setting. Black Sails (TV) is the precise hit — pirate
dynasty struggle. For films: Master and Commander has crew
dynamics but not family. There isn't a perfect movie match;
honest answer is "the film space is thin, here's the closest."

**With backing data:** Same two-axis pattern as #26 —
Succession's plot_analysis ("family power struggle, succession,
dynasty satire") AND theme = pirates / piracy. Crucially, the
result space might be empty in films-only, and the pipeline
should signal that rather than overreach.

---

## 28. Person-as-credit when role is wrong

**Query:** `starring wes anderson`

**Mental model:** Wes Anderson is a director, not an actor. So
either the user is confused or means Anderson's bit-part cameos
(he's appeared briefly in some Spike Jonze and Noah Baumbach
work). The reasonable read is "user wants Wes Anderson FILMS"
and the role marker is a mistake — Grand Budapest Hotel,
Moonrise Kingdom, Rushmore, The Royal Tenenbaums.

**With backing data:** Lexical lookup on "Wes Anderson" returns
him as director, not as cast. Soft intent correction → director
credit. The pipeline must handle role-marker mismatches without
returning empty (which would happen on strict actor=Anderson) or
silently dropping the role marker (which would scramble more
ambiguous queries).

---

## 29. Person-as-style — director as aesthetic reference

**Query:** `wes anderson does horror`

**Mental model:** Counterfactual style mash-up. Anderson hasn't
done horror — his oeuvre is whimsical melancholy with
symmetrical composition and pastel palettes. So either: a) what
WOULD a Wes Anderson horror film look like (style transfer
question), or b) Anderson-adjacent unsettling films. Closest
real picks: The Witch (period + meticulous aesthetic), The
Lighthouse (stylized, unsettling), Suspiria (Guadagnino has
visual flair), The Lobster (deadpan tone + dark themes).

**With backing data:** "Wes Anderson" here is a STYLE reference,
not a credit — needs to bind to `production` /
`narrative_techniques` (symmetrical composition,
pastel-whimsical-melancholy aesthetic) rather than director
credit. AND genre = horror. The composition is the test: does
the pipeline recognize that the same name maps to credit in
query #28 and to style in #29 based on context?

---

## 30. Hedged criterion — softener on genre

**Query:** `ideally a slow burn thriller`

**Mental model:** "Ideally" softens but doesn't remove the
criterion — user prefers slow-burn thriller but is open to
adjacent. Zodiac is the canonical slow-burn thriller, plus
Prisoners, The Vanishing (1988), Memories of Murder, A Most
Violent Year, The Conversation. Zodiac leads.

**With backing data:** Genre = thriller AND `viewer_experience`
= slow-burn pacing. The hedge softens the constraint slightly —
maybe broadening the threshold for inclusion — but the
criterion is still the dominant signal. The hedge is recorded
as a SOFTEN modal effect, not used as a reason to drop the
criterion entirely.

---

## 31. Hedged criterion — runtime only, no content

**Query:** `preferably under 2 hours`

**Mental model:** Constraint-only with a hedge, no content
signal. Like the negation-only queries, this needs a positive
ask before I can answer. If forced, I'd return well-regarded
sub-2-hour films across genres.

**With backing data:** Runtime < 120 min as a soft preference.
No positive content vector. Default surface = high-reception
under-runtime spread. Same shape as the negation-only queries:
the pipeline must recognize there's no positive signal and not
hallucinate one.

---

## 32. Dense — many wants stacked

**Query:** `90s sci fi action practical effects not too long good for friday night nothing depressing`

**Mental model:** Many criteria layered: 90s + sci-fi + action +
practical effects + reasonable runtime + Friday-night fun mood +
not depressing. Terminator 2 (1991) is the canonical answer —
hits every criterion. Total Recall (1990 borderline), Starship
Troopers (1997 — fun, practical effects), Predator 2 (1990),
The Fifth Element (1997). Excludes Children of Men (00s anyway,
but downbeat) and 12 Monkeys (depressing tone).

**With backing data:** Five-plus axis filter. Decade = 1990s,
genre = sci-fi + action, runtime < ~135 min, `viewer_experience`
for "fun, action-packed, not depressing," `production` for
"practical effects" (this last is the hard one — the production
vector covers technical achievements but "practical effects" is
a specific signal that may or may not be embedded). Reception
bias high. Tests the pipeline's ability to keep many soft
constraints alive without collapsing or dropping any.

---

## 33. Loose figurative — meta-reception

**Query:** `underrated`

**Mental model:** Films that didn't get the recognition they
deserved at release — quality but obscure or critically
overlooked. Brick (2005), Coherence, The Endless, The Vast of
Night, Margin Call, A Ghost Story, Sound of Metal (now better
known). The challenge is "underrated" is meta-reception, not a
content attribute — it's about the gap between quality and
notice.

**With backing data:** "Underrated" is a composite *signal*, not
a vector match. Possible proxies: high audience score with low
box office, high recent rating with low contemporary reception,
low TMDB popularity with high IMDB rating. This is a
ranker-level construct — the vectors don't know it directly. The
pipeline either needs an "underrated score" computed at ingest
or a runtime composite over reception + popularity gaps.

---

## 34. Loose figurative — gem framing

**Query:** `hidden gem`

**Mental model:** Same family as "underrated" with extra weight
on obscurity — films few have seen but are quality. Cult
favorites. Coherence, Primer, The Vast of Night, A Ghost Story,
Tucker and Dale vs Evil, Tigertail. The user wants the discovery
feeling more than the validation feeling.

**With backing data:** Same composite signal as #33, with
heavier weighting on low popularity and possibly cult-status
proxies (audience score >> popularity). The pipeline may need a
"hidden gem" precomputed score or a runtime weighting that
penalizes high-popularity films even when they have great
reception.

---

## 35. Parody-shape generalization — different anchor entity

**Query:** `spoof of marvel movies`

**Mental model:** Same idiom-shape as #6 (parody of the godfather)
but the reference is a franchise, not a single film. Hits: The
Boys (TV — pass), Mystery Men (1999), Sky High, Super (2010),
Defendor, The Specials. Mystery Men is the cleanest movie pick.
The user wants comedic deconstructions of superhero tropes, not
Marvel movies themselves and not generic comedies.

**With backing data:** The test is whether the pipeline treats
"spoof" as its own retrievable population (comedy/satire films
that target a genre) AND "marvel movies" as a reference shape
(superhero genre tropes, MCU-style). Independent retrievals
intersect; absorbing "spoof" as a modifier loses the comedic
requirement entirely. Genre = comedy/satire AND `plot_analysis` /
`production` referencing superhero tropes. Studio filter is
inverse — explicitly NOT Marvel.

---

## 36. Positioning-operator absorption that should NOT split

**Query:** `horror set in feudal japan`

**Mental model:** Onibaba (1964), Kuroneko (1968), Kwaidan (1965),
Throne of Blood (Macbeth-shaped, dark but not pure horror),
Hagazussa-adjacent. The "feudal japan" piece reshapes the horror
genre into a specific period setting; it doesn't independently
retrieve as a movie kind a user would ask for ("feudal japan"
alone is not a watching-want).

**With backing data:** "Set in feudal japan" is a true positioning
operator — it transposes the horror evaluation surface to a period
setting. Should absorb as a modifying_signal on "horror" rather
than emit as a peer atom. Genre = horror AND time period / setting
metadata = feudal japan (or `plot_events` signal for samurai-era
Japanese setting). The pipeline must distinguish this case (absorb
correct) from #35 (absorb wrong). Same surface shape, opposite
correct call.

---

## 37. Figurative population label + watch context

**Query:** `popcorn flick for friday night`

**Mental model:** "Popcorn flick" is a figurative label for a real
population — broad-appeal big-spectacle entertainment movies. Top
Gun: Maverick, Jurassic World, Mission Impossible series, the
Fast & Furious films, Mad Max: Fury Road. Plus the watching
context (Friday night = unwind, low cognitive load). Maverick is
the canonical recent answer.

**With backing data:** Tests whether the pipeline recognizes
figurative population labels (other than warm hug / feel good) as
their own carving criteria. `viewer_experience` for "fun,
spectacle, easy entertainment"; `watch_context` for "Friday night,
unwind." High reception bias. The risk is the pipeline treating
"popcorn flick" as a literal popcorn or food query, or stripping
it as filler.

---

## 38. Implicit-era constraint via "young"

**Query:** `young al pacino crime movie`

**Mental model:** "Young" implies pre-1990 Pacino. The Godfather
(1972), Serpico (1973), Dog Day Afternoon (1975), Scarface (1983),
Carlito's Way (1993 — borderline). The intersection of three
constraints (Pacino + young = early career window + crime genre)
is dense and high-quality.

**With backing data:** Three atoms: "al pacino" (lexical entity =
cast credit), "young" (implicit era constraint — needs to resolve
to a date range based on the named person's career arc), "crime
movie" (genre). The "young" piece is the interesting one — does
the pipeline absorb it as a modifying_signal that reshapes the
date filter on Pacino's filmography, or treat it as a vague hedge?
A correct read computes era-via-person rather than just hedging
the credit filter.

---

## 39. Meta-recognition + meta-quality qualifier

**Query:** `oscar bait but actually good`

**Mental model:** "Oscar bait" is a derogatory shorthand for
prestige films targeting awards via subject (illness, war,
biopic), gravitas, lead-acting showcase — but the user is asking
for the population of those films that genuinely deliver. Schindler's
List, There Will Be Blood, No Country for Old Men, Manchester by
the Sea, Moonlight, 12 Years a Slave. Excludes formulaic-but-
hollow prestige (think mid-tier Best Picture nominees).

**With backing data:** Two atoms: "oscar bait" (figurative
population — prestige-aimed dramas) and "actually good" (quality
qualifier ranging over the carved set). Tests award-as-population
recognition (does "oscar bait" route to award metadata + a
prestige-style cluster?) and tests whether quality language
("actually good") gets read as a high-reception qualifier rather
than treated as filler.

---

## 40. Genre-hybrid compound

**Query:** `musical horror`

**Mental model:** Sweeney Todd (2007), Repo! The Genetic Opera,
Little Shop of Horrors (1986), The Rocky Horror Picture Show,
Anna and the Apocalypse, The Lure (2015). Tiny intersection but a
real one. The user wants the hybrid, not "musicals OR horrors."

**With backing data:** The atomicity test in question — "musical
horror" reads as one compound atom (the population is films that
are BOTH musical and horror) rather than two parallel atoms whose
retrieval intersection might miss the named hybrid. Compare to "80s
action arnold" (#16), where the three pieces genuinely retrieve
and combine. Here, splitting risks returning generic horror or
generic musicals. Tests that the pipeline reads compound-genre
phrases as atoms when the population only exists at the
intersection.

---

## 41. Pure narrative-outcome description, no entity

**Query:** `movies where the villain wins`

**Mental model:** Se7en, No Country for Old Men, The Empire
Strikes Back, Chinatown, The Mist, Gone Girl, Prisoners (debatable),
The Cabin in the Woods. Films where the antagonist's project
succeeds or the protagonist meaningfully fails. Se7en and No
Country are the prototypes.

**With backing data:** Pure `plot_events` / `plot_analysis`
territory — narrative-outcome description with no lexical or
metadata anchor. Distinct shape from #7 (single trope: time loop)
in that "villain wins" is a structural plot property, not an event
or trope. Tests whether the vector spaces have enough resolution
on outcomes/endings vs general dark-tone or villain-centric films.
The risk is conflating "villain wins" with "villain is prominent"
or "ends darkly."

---

## 42. Cross-medium figurative comparison

**Query:** `feels like a video game`

**Mental model:** Films with video-game DNA — kinetic action
choreography, level-structure pacing, respawn/loop logic, or
explicit game adaptations done well. Edge of Tomorrow, John Wick
series, Crank, Hardcore Henry, Scott Pilgrim, Source Code, Free
Guy. Edge of Tomorrow is the canonical "lives like a video game
without being one." Excludes literal video-game adaptations
unless they happen to fit.

**With backing data:** "Video game" is not a movie population — it
names a non-film medium the user is using as a stylistic /
experiential reference. Tests how the pipeline handles
cross-medium analogy: does it route to `narrative_techniques` and
`viewer_experience` for game-like pacing/structure, or does it
collapse to a literal video-game-adaptation lexical lookup? The
right read carries the comparison through to vector space, not to
metadata.
