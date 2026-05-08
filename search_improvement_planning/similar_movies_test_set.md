# Similar Movies Test Set

This document enumerates the smoke-harness anchors and cohorts run by
[`search_v2/run_similar_movies_batch.py`](../search_v2/run_similar_movies_batch.py),
along with the films we expect to see hovering near the top of each
result set and *why*.

The "Expected near top" lists are not strict predictions — score
relativities depend on shape similarity, lane weights, multipliers,
and floors. They're the qualitative neighbors a thoughtful viewer
would surface, used as a calibration target when reading harness
output. The "Key signal" column names the lane(s) we expect to drive
the result.

## Single-anchor cases

### Inception (2010) — `27205`
**Active anchor types**: standard_shape, director_signature
**Key signal**: Nolan auteur (curated) + shape (puzzle / mind-bending)

| Expected near top | Why |
|---|---|
| Tenet | Nolan; time-inversion puzzle structure |
| The Prestige | Nolan; twist-driven, doppelgänger / identity |
| Memento | Nolan; memory / non-linear |
| Mulholland Drive | Lynch; dreamlike puzzle, identity |
| Eternal Sunshine of the Spotless Mind | Memory / dream-state shape |
| Trance | Hypnosis / dream-state heist |
| Vanilla Sky / Source Code | Reality-vs-dream puzzle |

### The Matrix (1999) — `603`
**Active anchor types**: standard_shape, franchise_dominant, source_material
**Key signal**: franchise (Matrix sequels) + shape (cyberpunk philosophy)

| Expected near top | Why |
|---|---|
| The Matrix Reloaded / Revolutions / Resurrections | Same franchise |
| Blade Runner / 2049 | Cyberpunk shape, reality-questioning |
| Dark City | Reality-as-construct, similar tone |
| Total Recall (1990) | Mind-vs-reality sci-fi |
| Inception | Adjacent (mind-bending sci-fi) |

### Star Wars (1977) — `11`
**Active anchor types**: standard_shape, prestige, franchise_dominant, studio_lineage
**Key signal**: franchise (saga) + studio + rare-keyword combo (jedi/force)

| Expected near top | Why |
|---|---|
| The Empire Strikes Back, Return of the Jedi | Same trilogy |
| The Force Awakens, Last Jedi, Rise of Skywalker | Sequel trilogy |
| Phantom Menace, Attack of the Clones, Revenge of the Sith | Prequel trilogy |
| Rogue One, Solo | Spinoff films, same lineage |

**Negative**: American Graffiti (Lucas, but not curated) — should NOT
fire on director lane (verifies V3 H7).

### Toy Story (1995) — `862`
**Active anchor types**: standard_shape, prestige, franchise_dominant, studio_lineage
**Key signal**: franchise (TS sequels) + studio (Pixar) + format (no shorts in top 5)

| Expected near top | Why |
|---|---|
| Toy Story 2 / 3 / 4 | Same franchise |
| Toy Story of Terror | Same franchise but a short — should appear at most once (V3 max-1 cap) |
| Monsters Inc, Inside Out, Up, Cars | Pixar studio |
| Finding Nemo | Pixar |

**Negative**: should NOT have a Pixar-shorts trio at slots 8/9/10 (V2
failure case, V3 H3 verifies fix).

### Spirited Away (2001) — `129`
**Active anchor types**: standard_shape, prestige, studio_lineage, director_signature
**Key signal**: Miyazaki auteur (curated) + studio (Ghibli)

| Expected near top | Why |
|---|---|
| My Neighbor Totoro, Princess Mononoke | Same studio, same director |
| Howl's Moving Castle, Castle in the Sky, Kiki's Delivery Service | Miyazaki Ghibli |
| Ponyo, The Wind Rises | Miyazaki Ghibli |

### The Godfather (1972) — `238`
**Active anchor types**: standard_shape, prestige, franchise_dominant
**Key signal**: franchise + prestige + crime-epic shape

| Expected near top | Why |
|---|---|
| The Godfather Part II / III | Same franchise |
| Goodfellas, Casino, The Irishman | Mob drama, prestige bucket |
| Once Upon a Time in America | Crime epic shape |
| Scarface | Crime epic shape |

### The Dark Knight (2008) — `155`
**Active anchor types**: standard_shape, prestige, franchise_dominant, director_signature
**Key signal**: franchise (Batman) + Nolan auteur

| Expected near top | Why |
|---|---|
| The Dark Knight Rises, Batman Begins | Same Nolan trilogy |
| Joker, The Batman (2022) | Same shared universe |
| Inception, Interstellar, Dunkirk, Tenet | Nolan auteur |
| Heat | Crime drama shape (Mann) |

### The Dark Knight Rises (2012) — `49026`
**Active anchor types**: standard_shape, prestige, franchise_dominant, director_signature
**Key signal**: franchise + Nolan auteur. **V3 H4 test**: zero animated Batman entries.

| Expected near top | Why |
|---|---|
| The Dark Knight, Batman Begins | Same Nolan trilogy |
| The Batman (2022), Batman 1989, Batman Returns | Live-action Batman |
| Other Nolan films | Auteur lane |

**Negative**: animated Batman (Year One, Long Halloween) should be
suppressed by the medium piecewise multiplier (cross-category 0.65×).

### Get Out (2017) — `419430`
**Active anchor types**: standard_shape, prestige, director_signature
**Key signal**: Peele auteur (curated) + social-horror shape

| Expected near top | Why |
|---|---|
| Us, Nope | Peele auteur |
| Antebellum, Candyman (2021) | Social-horror shape |
| The Babadook, Hereditary | Slow-burn psychological horror |

### John Wick (2014) — `245891`
**Active anchor types**: standard_shape, franchise_dominant
**Key signal**: franchise + neo-action shape

| Expected near top | Why |
|---|---|
| John Wick 2 / 3 / 4 | Same franchise |
| Atomic Blonde, Nobody, Bullet Train | Stylized neo-action |
| The Equalizer | Reluctant-killer revenge |

### Oppenheimer (2023) — `872585`
**Active anchor types**: standard_shape, prestige, source_material, director_signature
**Key signal**: Nolan auteur + prestige biopic + rare-keyword combo (Manhattan Project)

| Expected near top | Why |
|---|---|
| Dunkirk | Nolan; war/historical |
| Fat Man and Little Boy, Day One | Direct Manhattan-Project subject (rare-keyword) |
| Schindler's List, 12 Years a Slave, The Pianist | Prestige biopic shape |
| Nixon, JFK | Political biopic shape |
| The Imitation Game, A Beautiful Mind | Genius-scientist biopic |

**V3 H1 / H10 test**: top 5 contains zero documentaries; rare-keyword
lane fires meaningfully on the Manhattan-Project trait pool.

### Barbie (2023) — `346698`
**Active anchor types**: standard_shape, franchise_dominant, director_signature
**Key signal**: Gerwig auteur + female-led satire themes

| Expected near top | Why |
|---|---|
| Lady Bird, Little Women (2019) | Gerwig auteur (V3.1 director floor) |
| The Favourite | Female-led satire |
| Poor Things | Surreal female-led satire |
| Frances Ha, 20th Century Women | Female-led indie (Gerwig acted in Frances Ha) |
| Mean Girls, Legally Blonde, Clueless | Female-led satire / coming-of-age |
| I Am Not an Easy Man | Gender-flip comedy (themes match) |
| Where'd You Go, Bernadette | Female-led indie |

**V3.1 watch**: Lady Bird and The Favourite should enter top 10 via the
new single-anchor director floor + themes weight bump + themes recall.
**Negative**: top 1 should NOT be a Telugu / Bollywood film (V3 H5
test — "Swag" or similar should land mid-pack at best, not #1).

### Sharknado (2013) — `205321`
**Active anchor types**: standard_shape, cult_garbage, franchise_dominant
**Key signal**: cult_garbage quality bucket + franchise

| Expected near top | Why |
|---|---|
| Sharknado 2 / 3 / 4 / 5 / 6 | Same franchise |
| Mega Shark vs. Giant Octopus | Asylum studio, similar quality |
| Birdemic, Troll 2 | Cult-garbage tier |

### The Room (2003) — `17473`
**Active anchor types**: standard_shape, cult_garbage
**Key signal**: cult_garbage bucket + low-quality auteur (Wiseau)

| Expected near top | Why |
|---|---|
| The Disaster Artist | About The Room itself |
| Birdemic, Troll 2, Plan 9 from Outer Space | Cult-garbage tier |
| Manos: The Hands of Fate | Cult-bad classic |

### The Green Mile (1999) — `497`
**Active anchor types**: standard_shape, prestige, source_material
**Key signal**: Stephen King source + prestige bucket

| Expected near top | Why |
|---|---|
| The Shawshank Redemption | Same writer-director (Darabont), King adaptation |
| The Mist | Darabont + King |
| Stand By Me, Misery, Carrie, IT | King adaptations |
| The Shining | King adaptation |

**Note**: Darabont is not in the curated auteur list, so the director
lane is silent. Source-material lane (Stephen King novel) carries the
King connection. Shawshank is the canonical match — verify it's in
top 3.

### Pulp Fiction (1994) — `680`
**Active anchor types**: standard_shape, prestige, studio_lineage, director_signature
**Key signal**: Tarantino auteur (curated)

| Expected near top | Why |
|---|---|
| Reservoir Dogs, Jackie Brown, Kill Bill 1 / 2 | Tarantino |
| Inglourious Basterds, Django Unchained, Once Upon a Time in Hollywood, Hateful Eight | Tarantino |
| True Romance | Tarantino-written |
| Sin City | Stylized crime-anthology |

### Fight Club (1999) — `550`
**Active anchor types**: standard_shape, studio_lineage, source_material, director_signature
**Key signal**: Fincher auteur (curated)

| Expected near top | Why |
|---|---|
| Se7en, Zodiac, Gone Girl | Fincher |
| The Game | Fincher |
| American Psycho | Adjacent (anti-consumerist male-id satire) |
| Memento | Adjacent psychological thriller |

### Back to the Future (1985) — `105`
**Active anchor types**: standard_shape, prestige, franchise_dominant
**Key signal**: franchise + 80s-sci-fi-comedy shape

| Expected near top | Why |
|---|---|
| Back to the Future Part II / III | Same franchise |
| Bill & Ted's Excellent Adventure | Time-travel comedy |
| Project Almanac, Hot Tub Time Machine | Time-travel ensembles |
| Looper | Time-travel adjacent |

### Interstellar (2014) — `157336`
**Active anchor types**: standard_shape, director_signature
**Key signal**: Nolan auteur + space-drama shape

| Expected near top | Why |
|---|---|
| Inception, Tenet, Dunkirk, The Prestige, Memento | Nolan |
| Gravity, Ad Astra, Contact, The Martian | Space-drama shape |
| 2001: A Space Odyssey | Foundational adjacent |

### LOTR: Fellowship of the Ring (2001) — `120`
**Active anchor types**: standard_shape, prestige, franchise_dominant, studio_lineage, source_material
**Key signal**: franchise + Tolkien source + epic-fantasy shape

| Expected near top | Why |
|---|---|
| The Two Towers, Return of the King | Same franchise |
| The Hobbit (3 films) | Same lineage, Tolkien source |
| Other epic fantasy adaptations | Source/shape |

### Titanic (1997) — `597`
**Active anchor types**: standard_shape
**Key signal**: shape (epic romance/disaster). Cameron is NOT curated.

| Expected near top | Why |
|---|---|
| Titanic (1953), A Night to Remember, S.O.S. Titanic | Same subject |
| The English Patient, Atonement | Epic period romance |
| Pearl Harbor | Epic disaster-romance shape |

## Multi-anchor cohorts

### Nolan trio — `(27205, 77, 1124)`
**Anchors**: Inception, Memento, The Prestige
**Key signal**: Nolan auteur on all 3 (M_d/N = 1.0) → max director contribution; multi-anchor director floor at 0.35; thematic cohesion (puzzles, identity, non-linear narrative)

| Expected near top | Why |
|---|---|
| Tenet, Interstellar, Dunkirk, The Dark Knight | Other Nolan films |
| Mulholland Drive, Memento-adjacent puzzles | Shape + themes |
| Eternal Sunshine of the Spotless Mind | Themes (memory) |

### Pixar trio — `(862, 12, 14160)`
**Anchors**: Toy Story, Finding Nemo, Up
**Key signal**: studio (Pixar M_d/N = 1.0) + family-animation shape

| Expected near top | Why |
|---|---|
| Toy Story 2 / 3 / 4, Monsters Inc, Inside Out | Pixar |
| Coco, Soul, Wall-E, The Incredibles | Pixar |
| Cars, Ratatouille, Onward | Pixar |

### Ghibli trio — `(129, 8392, 128)`
**Anchors**: Spirited Away, My Neighbor Totoro, Princess Mononoke
**Key signal**: studio (Ghibli) + Miyazaki auteur (curated, M_d/N = 1.0)

| Expected near top | Why |
|---|---|
| Howl's Moving Castle, Kiki's Delivery Service, Castle in the Sky, Ponyo | Same studio + director |
| Grave of the Fireflies | Ghibli (Takahata) |
| The Wind Rises | Miyazaki |

### MCU trio — `(1726, 24428, 271110)`
**Anchors**: Iron Man, The Avengers, Captain America: Civil War
**Key signal**: franchise (MCU) + studio (Marvel) + cast (RDJ in 2/3 anchors) + source (Marvel comics)

| Expected near top | Why |
|---|---|
| Iron Man 2 / 3, Avengers Age of Ultron, Infinity War, Endgame | MCU mainline |
| Captain America Winter Soldier / First Avenger, Cap: Brave New World | MCU |
| Guardians of the Galaxy, Thor, Black Panther | MCU |

### Stephen King horror — `(694, 346364, 235)`
**Anchors**: The Shining, IT, Misery
**Key signal**: source (Stephen King novels, M_t/N = 1.0) + horror shape

| Expected near top | Why |
|---|---|
| Carrie, The Mist, Pet Sematary, 1408 | King adaptations |
| Cujo, Christine, Salem's Lot | King adaptations |
| The Shining (TV), IT Chapter Two | King |

### Best Picture trio — `(238, 424, 76203)`
**Anchors**: The Godfather, Schindler's List, 12 Years a Slave
**Key signal**: prestige bucket + format (narrative feature) + source-material co-occurrence

| Expected near top | Why |
|---|---|
| The Pianist, The Killing Fields, The Mission | Prestige biopic shape |
| Citizen Kane, The Grapes of Wrath, Killers of the Flower Moon | Prestige American epic |
| The Godfather Part II | Same franchise as one anchor |

**V3 H1 test**: zero documentaries in top 5.

### Tarantino trio — `(680, 24, 273248)`
**Anchors**: Pulp Fiction, Kill Bill 1, The Hateful Eight
**Key signal**: Tarantino auteur on all 3 (M_d/N = 1.0)

| Expected near top | Why |
|---|---|
| Reservoir Dogs, Jackie Brown, Inglourious Basterds, Once Upon a Time in Hollywood, Django Unchained | Tarantino |
| Kill Bill 2 | Same lineage as one anchor |
| Sin City, True Romance | Tarantino-adjacent (writer credit / similar style) |

### Spielberg adventure trio — `(1894, 89, 329)`
**Anchors**: Indiana Jones (Raiders, Last Crusade, Lost World? — IDs map to Spielberg adventure films)
**Key signal**: franchise (Indy) + adventure-shape. Spielberg is NOT in curated auteur list, so director lane is silent.

| Expected near top | Why |
|---|---|
| Other Indy films | Same franchise |
| The Mummy (1999), Romancing the Stone, National Treasure | Adventure shape |
| Tomb Raider | Adventure shape |

### WW2 epics — `(424, 857, 562)`
**Anchors**: Schindler's List, Saving Private Ryan, Das Boot
**Key signal**: themes (WW2 / war) + prestige + format

| Expected near top | Why |
|---|---|
| The Pianist, Defiance, Inglourious Basterds | WW2 themes |
| The Thin Red Line, Letters from Iwo Jima, Flags of Our Fathers | WW2 |
| Atonement, Dunkirk | WW2 adjacent |

### Slasher trio — `(1091, 9716, 4233)`
**Anchors**: The Thing, Carrie, Halloween
**Key signal**: themes (horror / slasher) + format

| Expected near top | Why |
|---|---|
| Friday the 13th, A Nightmare on Elm Street, Scream | Slasher canon |
| The Texas Chain Saw Massacre, Hellraiser | Body-horror canon |
| Other Carpenter films (The Fog, They Live) | Carpenter (curated) auteur |

### Romcom trio — `(114, 1581, 639)`
**Anchors**: Pretty Woman, The Holiday, When Harry Met Sally
**Key signal**: format + themes (romcom)

| Expected near top | Why |
|---|---|
| Notting Hill, You've Got Mail, Sleepless in Seattle | Same era romcoms |
| 10 Things I Hate About You, How to Lose a Guy in 10 Days | Romcom canon |
| Bridget Jones's Diary, Love Actually | Romcom canon |

### Studio Ghibli + Pixar mix — `(862, 129, 8392)`
**Anchors**: Toy Story, Spirited Away, My Neighbor Totoro
**Key signal**: cross-tradition cohesion test. Studio cohesion is split (Pixar vs. Ghibli, neither M_t/N ≥ 0.67); director cohesion is split (Lasseter not curated, Miyazaki curated only on 2/3). Tests the V3 low-cohesion fallback or broad family-animation themes.

| Expected near top | Why |
|---|---|
| The Iron Giant, Coraline, Wall-E | Family animation, broad shape |
| Other Ghibli + Pixar mix candidates | Animation shape |

May trigger low-cohesion fallback to round-robin per-anchor.

### Tom Hanks trio (H9) — `(2280, 13, 862)`
**Anchors**: Big, Forrest Gump, Toy Story
**Key signal**: cast lane — Hanks in top-3 billing of all 3 anchors (M_a/N = 1.0); cast floor at 0.45

| Expected near top | Why |
|---|---|
| Toy Story 3 / 2, The Terminal, Cast Away, Captain Phillips, Apollo 13, Sleepless in Seattle | Hanks vehicles |
| The Polar Express, Saving Private Ryan, Sully | Hanks vehicles |
| The Iron Giant, E.T., Christopher Robin | Family / coming-of-age shape |

### Female-led / Gerwig — `(346698, 391713, 331482)`
**Anchors**: Barbie, Lady Bird, Little Women
**Key signal**: Gerwig directs all 3 anchors; the lane fires only for *other* Gerwig films (none in catalog beyond the 3 anchors). Falls back to themes (female-led, coming-of-age) and shape.

| Expected near top | Why |
|---|---|
| 20th Century Women, The Favourite, Where'd You Go Bernadette | Female-led indie themes |
| Frances Ha (Baumbach + Gerwig writer/star) | Adjacent |
| Fried Green Tomatoes, Circle of Friends | Female-ensemble drama |
| Rushmore, The Fabelmans | Coming-of-age shape |

This cohort tests V3.1 themes recall + themes weight bump in a case
where the director lane is structurally silent.

## How to use this list

1. Run `python -m search_v2.run_similar_movies_batch --multi --limit 10`.
2. For each anchor / cohort, scan the top 10 against the "Expected near
   top" list.
3. If an expected film is missing, check the per-row breakdown in
   the markdown output:
   - Was it in the candidate pool at all? (Lanes column shows what
     surfaced it.)
   - Did the lane fire as expected? (`raw→contribution` makes it
     visible.)
   - Was a multiplier suppressing it? (Multipliers row.)
4. If an unexpected film is at #1 or #2, check whether a single lane
   is dominating (`dominant_lane`) and whether the multiplier /
   floor stack pushed it there.
5. The `lanes` column should generally include `themes, rare_keyword`
   for thematic-driven anchors and `franchise, studio` for
   franchise/studio-driven anchors. Missing those is a signal the
   recall path or the lane scoring isn't engaging.
