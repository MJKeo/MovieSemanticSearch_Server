# Similar Movies — Multi-Anchor Test Groups

## Nolan trio (director-heavy)

_Inception, Memento, The Prestige_

Anchors: Inception (2010) `27205`, Memento (2000) `77`, The Prestige (2006) `1124`

Active anchor types: standard_shape

Lane weights (normalized): shape=0.714, director=0.286, franchise=0.000, studio=0.000, source=0.000, quality=0.000
Raw lane weights:           shape=0.600, director=0.240, franchise=0.000, studio=0.000, source=0.000, quality=0.000

Vector-space cohesion: anchor=0.621, plot_events=0.168, plot_analysis=0.408, viewer_experience=1.000, watch_context=1.000, narrative_techniques=0.988, production=0.100, reception=0.755
Vector-space weights:  anchor=0.123, plot_events=0.042, plot_analysis=0.109, viewer_experience=0.196, watch_context=0.179, narrative_techniques=0.162, production=0.046, reception=0.143

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Mulholland Drive (2001) `1018` | 0.714 | shape | shape | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | Shutter Island (2010) `11324` | 0.651 | shape | shape | 0.91 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | Trance (2013) `68727` | 0.610 | shape | shape | 0.85 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | Chaos (2000) `39983` | 0.570 | shape | shape | 0.80 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | Buster's Mal Heart (2017) `367147` | 0.549 | shape | shape | 0.77 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | The I Inside (2004) `11588` | 0.513 | shape | shape | 0.72 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | Mentally Apart (2020) `799128` | 0.496 | shape | shape | 0.69 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | I'm Thinking of Ending Things (2020) `500840` | 0.486 | shape | shape | 0.68 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | Following (1999) `11660` | 0.448 | director | shape, director | 0.23 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | Epilog (1992) `85923` | 0.447 | shape | shape | 0.63 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Pixar trio (studio + family animation)

_Toy Story, Finding Nemo, Up_

Anchors: Toy Story (1995) `862`, Finding Nemo (2003) `12`, Up (2009) `14160`

Active anchor types: standard_shape, studio_lineage, prestige

Lane weights (normalized): shape=0.714, director=0.000, franchise=0.000, studio=0.143, source=0.000, quality=0.143
Raw lane weights:           shape=0.600, director=0.000, franchise=0.000, studio=0.120, source=0.000, quality=0.120

Vector-space cohesion: anchor=0.593, plot_events=0.172, plot_analysis=0.472, viewer_experience=1.000, watch_context=1.000, narrative_techniques=0.899, production=1.000, reception=0.781
Vector-space weights:  anchor=0.105, plot_events=0.037, plot_analysis=0.105, viewer_experience=0.173, watch_context=0.158, narrative_techniques=0.132, production=0.158, reception=0.130

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Toy Story 3 (2010) `10193` | 1.000 | shape | shape, studio, quality | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| 2 | Inside Out (2015) `150540` | 0.834 | shape | shape, studio, quality | 0.77 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| 3 | Monsters, Inc. (2001) `585` | 0.720 | shape | shape, studio | 0.81 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 4 | Cars (2006) `920` | 0.661 | shape | shape, studio | 0.73 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 5 | Toy Story 4 (2019) `301528` | 0.614 | shape | shape, studio | 0.66 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 6 | Toy Story 2 (1999) `863` | 0.611 | shape | shape, studio | 0.65 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 7 | Ratatouille (2007) `2062` | 0.605 | shape | shape, studio, quality | 0.45 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| 8 | The Good Dinosaur (2015) `105864` | 0.589 | shape | shape, studio | 0.62 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 9 | Onward (2020) `508439` | 0.578 | shape | shape, studio | 0.61 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 10 | A Bug's Life (1998) `9487` | 0.523 | shape | shape, studio | 0.53 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |

## Ghibli / Miyazaki trio

_Spirited Away, My Neighbor Totoro, Princess Mononoke_

Anchors: Spirited Away (2001) `129`, My Neighbor Totoro (1988) `8392`, Princess Mononoke (1997) `128`

Active anchor types: standard_shape, studio_lineage

Lane weights (normalized): shape=0.625, director=0.250, franchise=0.000, studio=0.125, source=0.000, quality=0.000
Raw lane weights:           shape=0.600, director=0.240, franchise=0.000, studio=0.120, source=0.000, quality=0.000

Vector-space cohesion: anchor=0.677, plot_events=0.253, plot_analysis=0.569, viewer_experience=1.000, watch_context=0.945, narrative_techniques=0.888, production=1.000, reception=0.794
Vector-space weights:  anchor=0.113, plot_events=0.047, plot_analysis=0.114, viewer_experience=0.169, watch_context=0.147, narrative_techniques=0.127, production=0.154, reception=0.128

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Ponyo (2008) `12429` | 1.000 | shape | shape, director, studio | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 2 | Howl's Moving Castle (2004) `4935` | 0.917 | shape | shape, director, studio | 0.87 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 3 | Castle in the Sky (1986) `10515` | 0.752 | shape | shape, director, studio | 0.60 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 4 | The Boy and the Heron (2023) `508883` | 0.665 | shape | shape, director, studio | 0.46 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 5 | Porco Rosso (1992) `11621` | 0.644 | shape | shape, director, studio | 0.43 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 6 | The Tale of The Princess Kaguya (2013) `149871` | 0.625 | shape | shape, studio | 0.80 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 7 | Kiki's Delivery Service (1989) `16859` | 0.606 | director | shape, director, studio | 0.37 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 8 | The Wind Rises (2013) `149870` | 0.602 | director | shape, director, studio | 0.36 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 9 | The Secret World of Arrietty (2010) `51739` | 0.533 | shape | shape, studio | 0.65 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 10 | Mei and the Kittenbus (2002) `158483` | 0.463 | director | shape, director, studio | 0.14 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |

## MCU trio (franchise-shared-universe)

_Iron Man, The Avengers, Captain America: Civil War_

Anchors: Iron Man (2008) `1726`, The Avengers (2012) `24428`, Captain America: Civil War (2016) `271110`

Active anchor types: standard_shape, franchise_dominant, studio_lineage, source_material

Lane weights (normalized): shape=0.577, director=0.000, franchise=0.231, studio=0.115, source=0.077, quality=0.000
Raw lane weights:           shape=0.600, director=0.000, franchise=0.240, studio=0.120, source=0.080, quality=0.000

Vector-space cohesion: anchor=0.701, plot_events=0.355, plot_analysis=0.583, viewer_experience=1.000, watch_context=0.911, narrative_techniques=0.782, production=0.100, reception=0.898
Vector-space weights:  anchor=0.129, plot_events=0.067, plot_analysis=0.129, viewer_experience=0.188, watch_context=0.159, narrative_techniques=0.127, production=0.045, reception=0.157

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Avengers: Age of Ultron (2015) `99861` | 1.000 | shape | shape, franchise, studio, source | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| 2 | Iron Man 2 (2010) `10138` | 0.946 | shape | shape, franchise, studio, source | 0.91 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| 3 | Captain America: The Winter Soldier (2014) `100402` | 0.760 | shape | shape, franchise, studio, source | 0.58 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| 4 | X2 (2003) `36658` | 0.397 | shape | shape, source | 0.56 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| 5 | Iron Man 3 (2013) `68721` | 0.754 | shape | shape, franchise, studio, source | 0.57 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| 6 | Guardians of the Galaxy (2014) `118340` | 0.710 | shape | shape, franchise, studio, source | 0.50 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| 7 | Spider-Man: Far From Home (2019) `429617` | 0.656 | shape | shape, franchise, studio, source | 0.40 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| 8 | Captain America: Brave New World (2025) `822119` | 0.645 | franchise | shape, franchise, studio, source | 0.39 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| 9 | Avengers: Infinity War (2018) `299536` | 0.596 | franchise | shape, franchise, studio, source | 0.30 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| 10 | Captain America: The First Avenger (2011) `1771` | 0.592 | franchise | shape, franchise, studio, source | 0.29 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 |

## Stephen King horror (source material)

_The Shining, IT (2017), Misery_

Anchors: The Shining (1980) `694`, It (2017) `346364`, Misery (1990) `1700`

Active anchor types: standard_shape, source_material

Lane weights (normalized): shape=0.882, director=0.000, franchise=0.000, studio=0.000, source=0.118, quality=0.000
Raw lane weights:           shape=0.600, director=0.000, franchise=0.000, studio=0.000, source=0.080, quality=0.000

Vector-space cohesion: anchor=0.399, plot_events=0.100, plot_analysis=0.295, viewer_experience=1.000, watch_context=0.954, narrative_techniques=0.875, production=0.100, reception=0.809
Vector-space weights:  anchor=0.098, plot_events=0.034, plot_analysis=0.100, viewer_experience=0.212, watch_context=0.186, narrative_techniques=0.157, production=0.050, reception=0.163

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Ghost Story (1981) `24634` | 0.890 | shape | shape, source | 0.88 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| 2 | 1408 (2007) `3021` | 0.882 | shape | shape | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | Veronica (2017) `441701` | 0.805 | shape | shape | 0.91 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | The Forbidden Door (2009) `39024` | 0.797 | shape | shape, source | 0.77 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| 5 | Nocturnal Animals (2016) `340666` | 0.771 | shape | shape, source | 0.74 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| 6 | In Our Blood (2025) `1312228` | 0.769 | shape | shape | 0.87 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | The Innocents (2021) `660942` | 0.740 | shape | shape | 0.84 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | The Year of Fury (2021) `717689` | 0.737 | shape | shape, source | 0.70 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| 9 | Caveat (2021) `744746` | 0.735 | shape | shape | 0.83 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | The Collector (1965) `42740` | 0.733 | shape | shape, source | 0.70 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |

## Best Picture prestige drama (no shared metadata)

_The Godfather, Schindler's List, 12 Years a Slave_

Anchors: The Godfather (1972) `238`, Schindler's List (1993) `424`, 12 Years a Slave (2013) `76203`

Active anchor types: standard_shape, source_material, prestige

Lane weights (normalized): shape=0.750, director=0.000, franchise=0.000, studio=0.000, source=0.100, quality=0.150
Raw lane weights:           shape=0.600, director=0.000, franchise=0.000, studio=0.000, source=0.080, quality=0.120

Vector-space cohesion: anchor=0.359, plot_events=0.100, plot_analysis=0.100, viewer_experience=1.000, watch_context=0.846, narrative_techniques=0.737, production=0.100, reception=0.587
Vector-space weights:  anchor=0.103, plot_events=0.039, plot_analysis=0.077, viewer_experience=0.238, watch_context=0.190, narrative_techniques=0.152, production=0.057, reception=0.144

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | The Pianist (2002) `423` | 0.972 | shape | shape, source, quality | 0.96 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| 2 | The Shop on Main Street (1965) `25905` | 0.750 | shape | shape | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | The Grapes of Wrath (1940) `596` | 0.726 | shape | shape, source, quality | 0.63 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| 4 | The Killing Fields (1984) `625` | 0.721 | shape | shape, source | 0.87 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 |
| 5 | Katyn (2007) `13614` | 0.717 | shape | shape, source | 0.87 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 |
| 6 | The Mission (1986) `11416` | 0.699 | shape | shape, source | 0.84 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 |
| 7 | Citizen Kane (1941) `15` | 0.666 | shape | shape, quality | 0.69 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| 8 | Holy Cross (2003) `438345` | 0.659 | shape | shape | 0.88 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | The Godfather Part II (1974) `240` | 0.647 | shape | shape, source, quality | 0.53 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| 10 | The Pawnbroker (1965) `20540` | 0.617 | shape | shape, source | 0.69 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |

## Cult-garbage trio

_Sharknado, The Room, Birdemic_

Anchors: Sharknado (2013) `205321`, The Room (2003) `17473`, Birdemic: Shock and Terror (2010) `40016`

Active anchor types: standard_shape, cult_garbage

Lane weights (normalized): shape=0.833, director=0.000, franchise=0.000, studio=0.000, source=0.000, quality=0.167
Raw lane weights:           shape=0.600, director=0.000, franchise=0.000, studio=0.000, source=0.000, quality=0.120

Vector-space cohesion: anchor=0.506, plot_events=0.100, plot_analysis=0.204, viewer_experience=1.000, watch_context=1.000, narrative_techniques=0.669, production=0.100, reception=0.854
Vector-space weights:  anchor=0.117, plot_events=0.035, plot_analysis=0.087, viewer_experience=0.215, watch_context=0.196, narrative_techniques=0.127, production=0.051, reception=0.173

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Troll 2 (1990) `26914` | 1.000 | shape | shape, quality | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| 2 | Space Mutiny (1988) `32148` | 0.817 | shape | shape, quality | 0.83 | 0.00 | 0.00 | 0.00 | 0.00 | 0.76 |
| 3 | Plan 9 from Outer Space (1957) `10513` | 0.681 | shape | shape | 0.82 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | Birdemic 2: The Resurrection (2013) `188489` | 0.625 | shape | shape, quality | 0.61 | 0.00 | 0.00 | 0.00 | 0.00 | 0.69 |
| 5 | Mega Shark vs. Giant Octopus (2009) `17911` | 0.554 | shape | shape, quality | 0.46 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| 6 | Sharknado 4: The 4th Awakens (2016) `390989` | 0.530 | shape | shape | 0.64 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | Mayday (2019) `626576` | 0.513 | shape | shape | 0.62 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | Hard Rock Zombies (1985) `28128` | 0.489 | shape | shape | 0.59 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | Airplane vs Volcano (2014) `258210` | 0.453 | shape | shape | 0.54 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | Clerk (1989) `683193` | 0.446 | shape | shape | 0.54 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Tarantino trio (director + studio)

_Pulp Fiction, Reservoir Dogs, Kill Bill Vol 1_

Anchors: Pulp Fiction (1994) `680`, Reservoir Dogs (1992) `500`, Kill Bill: Vol. 1 (2003) `24`

Active anchor types: standard_shape, studio_lineage

Lane weights (normalized): shape=0.637, director=0.255, franchise=0.000, studio=0.108, source=0.000, quality=0.000
Raw lane weights:           shape=0.600, director=0.240, franchise=0.000, studio=0.101, source=0.000, quality=0.000

Vector-space cohesion: anchor=0.546, plot_events=0.393, plot_analysis=0.337, viewer_experience=1.000, watch_context=0.702, narrative_techniques=0.696, production=0.323, reception=0.748
Vector-space weights:  anchor=0.117, plot_events=0.078, plot_analysis=0.103, viewer_experience=0.205, watch_context=0.141, narrative_techniques=0.125, production=0.083, reception=0.148

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Kill Bill: Vol. 2 (2004) `393` | 0.749 | shape | shape, director, studio | 0.66 | 1.00 | 0.00 | 0.67 | 0.00 | 0.00 |
| 2 | True Romance (1993) `319` | 0.699 | shape | shape, studio | 0.98 | 0.00 | 0.00 | 0.67 | 0.00 | 0.00 |
| 3 | Caliber 9 (1972) `40022` | 0.637 | shape | shape | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | Tazza: The High Rollers (2006) `38015` | 0.608 | shape | shape | 0.95 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | Sin City (2005) `187` | 0.542 | shape | shape, director | 0.45 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | Inglourious Basterds (2009) `16869` | 0.532 | shape | shape, director | 0.43 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | The Hateful Eight (2015) `273248` | 0.468 | director | shape, director | 0.33 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | Truth or Consequences, N.M. (1997) `31017` | 0.459 | shape | shape | 0.72 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | Jackie Brown (1997) `184` | 0.454 | director | shape, director, studio | 0.20 | 1.00 | 0.00 | 0.67 | 0.00 | 0.00 |
| 10 | Lucky Number Slevin (2006) `186` | 0.451 | shape | shape | 0.71 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Heist trio (vector-only, no metadata cohesion)

_Ocean's Eleven, Heat, The Italian Job_

Anchors: Ocean's Eleven (2001) `161`, Heat (1995) `949`, The Italian Job (2003) `9654`

Active anchor types: standard_shape, source_material

Lane weights (normalized): shape=0.899, director=0.000, franchise=0.000, studio=0.000, source=0.101, quality=0.000
Raw lane weights:           shape=0.600, director=0.000, franchise=0.000, studio=0.000, source=0.068, quality=0.000

Vector-space cohesion: anchor=0.555, plot_events=0.100, plot_analysis=0.806, viewer_experience=1.000, watch_context=0.848, narrative_techniques=1.000, production=0.100, reception=0.581
Vector-space weights:  anchor=0.114, plot_events=0.032, plot_analysis=0.169, viewer_experience=0.197, watch_context=0.158, narrative_techniques=0.165, production=0.047, reception=0.118

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Ocean's Thirteen (2007) `298` | 0.899 | shape | shape | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | Players (2012) `83382` | 0.861 | shape | shape, source | 0.88 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 |
| 3 | Foolproof (2003) `14527` | 0.856 | shape | shape | 0.95 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | The Italian Job (1969) `10536` | 0.826 | shape | shape | 0.92 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | Fast Five (2011) `51497` | 0.818 | shape | shape | 0.91 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | Lift (2024) `955916` | 0.761 | shape | shape | 0.85 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | Den of Thieves (2018) `449443` | 0.704 | shape | shape | 0.78 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | The Thieves (2012) `124157` | 0.684 | shape | shape | 0.76 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | Takers (2010) `22907` | 0.681 | shape | shape | 0.76 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | The Rise (2012) `128241` | 0.673 | shape | shape | 0.75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## LotR trilogy (everything cohesive)

_LotR: Fellowship / Two Towers / Return of the King_

Anchors: The Lord of the Rings: The Fellowship of the Ring (2001) `120`, The Lord of the Rings: The Two Towers (2002) `121`, The Lord of the Rings: The Return of the King (2003) `122`

Active anchor types: standard_shape, franchise_dominant, studio_lineage, source_material, prestige

Lane weights (normalized): shape=0.429, director=0.171, franchise=0.171, studio=0.086, source=0.057, quality=0.086
Raw lane weights:           shape=0.600, director=0.240, franchise=0.240, studio=0.120, source=0.080, quality=0.120

Vector-space cohesion: anchor=0.982, plot_events=0.872, plot_analysis=0.975, viewer_experience=1.000, watch_context=1.000, narrative_techniques=0.929, production=1.000, reception=1.000
Vector-space weights:  anchor=0.126, plot_events=0.104, plot_analysis=0.137, viewer_experience=0.140, watch_context=0.128, narrative_techniques=0.110, production=0.128, reception=0.128

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | The Hobbit: An Unexpected Journey (2012) `49051` | 0.837 | shape | shape, director, franchise, studio, source | 0.82 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| 2 | The Hobbit: The Battle of the Five Armies (2014) `122917` | 0.816 | shape | shape, director, franchise, studio, source | 0.77 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| 3 | The Hobbit: The Desolation of Smaug (2013) `57158` | 0.679 | shape | shape, director, studio, source | 0.85 | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 |
| 4 | The Lord of the Rings (1978) `123` | 0.657 | shape | shape, franchise, source | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| 5 | The Return of the King (1980) `1361` | 0.424 | shape | shape, franchise, source | 0.46 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| 6 | The Lord of the Rings: The War of the Rohirrim (2024) `839033` | 0.382 | franchise | shape, franchise, studio, source | 0.16 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| 7 | King Kong (2005) `254` | 0.335 | director | shape, director | 0.38 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | Born of Hope (2009) `1576537` | 0.313 | franchise | shape, franchise, source | 0.20 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| 9 | The Dark Knight Rises (2012) `49026` | 0.305 | shape | shape | 0.71 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | Harry Potter and the Deathly Hallows: Part 2 (2011) `12445` | 0.300 | shape | shape, source | 0.57 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |

## Chaotic mixed bag (intentionally incohesive)

_Toy Story, The Godfather, Sharknado_

Anchors: Toy Story (1995) `862`, The Godfather (1972) `238`, Sharknado (2013) `205321`

Active anchor types: standard_shape, prestige

Lane weights (normalized): shape=0.855, director=0.000, franchise=0.000, studio=0.000, source=0.000, quality=0.145
Raw lane weights:           shape=0.600, director=0.000, franchise=0.000, studio=0.000, source=0.000, quality=0.101

Vector-space cohesion: anchor=0.100, plot_events=0.100, plot_analysis=0.100, viewer_experience=0.963, watch_context=0.206, narrative_techniques=0.531, production=0.100, reception=0.100
Vector-space weights:  anchor=0.080, plot_events=0.055, plot_analysis=0.109, viewer_experience=0.327, watch_context=0.106, narrative_techniques=0.163, production=0.080, reception=0.080

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | A Bronx Tale (1993) `1607` | 0.855 | shape | shape | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | Toy Story 3 (2010) `10193` | 0.753 | shape | shape, quality | 0.77 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| 3 | Life or Something Like It (2002) `16643` | 0.741 | shape | shape | 0.87 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | Star Wars (1977) `11` | 0.619 | shape | shape, quality | 0.61 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| 5 | Mission to Mars (2000) `2067` | 0.598 | shape | shape | 0.70 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | Star Trek III: The Search for Spock (1984) `157` | 0.564 | shape | shape | 0.66 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | Phantom of the Opera (1943) `15855` | 0.500 | shape | shape | 0.58 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | Sharknado 4: The 4th Awakens (2016) `390989` | 0.464 | shape | shape | 0.54 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | Sword in the Desert (1949) `293258` | 0.463 | shape | shape | 0.54 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | GoodFellas (1990) `769` | 0.436 | shape | shape, quality | 0.40 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 |

## War film trio (prestige + vector)

_Saving Private Ryan, 1917, Apocalypse Now_

Anchors: Saving Private Ryan (1998) `857`, 1917 (2019) `530915`, Apocalypse Now (1979) `28`

Active anchor types: standard_shape, prestige

Lane weights (normalized): shape=0.855, director=0.000, franchise=0.000, studio=0.000, source=0.000, quality=0.145
Raw lane weights:           shape=0.600, director=0.000, franchise=0.000, studio=0.000, source=0.000, quality=0.101

Vector-space cohesion: anchor=0.519, plot_events=0.123, plot_analysis=0.614, viewer_experience=1.000, watch_context=0.929, narrative_techniques=0.764, production=0.100, reception=0.927
Vector-space weights:  anchor=0.109, plot_events=0.036, plot_analysis=0.141, viewer_experience=0.198, watch_context=0.170, narrative_techniques=0.131, production=0.047, reception=0.170

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | All Quiet on the Western Front (2022) `49046` | 0.855 | shape | shape | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | Platoon (1986) `792` | 0.650 | shape | shape, quality | 0.65 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| 3 | Dunkirk (2017) `374720` | 0.603 | shape | shape, quality | 0.59 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| 4 | All Quiet on the Western Front (1930) `143` | 0.517 | shape | shape, quality | 0.49 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| 5 | Wooden Crosses (1932) `32859` | 0.510 | shape | shape | 0.60 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | The Thin Red Line (1964) `188608` | 0.500 | shape | shape | 0.58 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | Tae Guk Gi: The Brotherhood of War (2004) `11658` | 0.479 | shape | shape | 0.56 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | All the King's Men (1999) `53253` | 0.470 | shape | shape | 0.55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | Kajaki (2014) `306650` | 0.469 | shape | shape | 0.55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | Saints and Soldiers (2003) `10105` | 0.445 | shape | shape | 0.52 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
