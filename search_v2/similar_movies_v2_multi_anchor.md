# Similar Movies V2 — Multi-Anchor Test Groups

## Nolan trio (director-heavy)

Anchors: Inception (2010) `27205`, Memento (2000) `77`, The Prestige (2006) `1124`

Active anchor types: standard_shape

Lane weights (normalized): shape=0.658, director=0.158, franchise=0.000, studio=0.000, source=0.000, quality=0.000, format=0.053, themes=0.079, cast=0.000, specific_award=0.053
Raw lane weights:           shape=0.998, director=0.240, franchise=0.000, studio=0.000, source=0.000, quality=0.000, format=0.080, themes=0.120, cast=0.000, specific_award=0.080

Vector-space cohesion: anchor=0.621, plot_events=0.168, plot_analysis=0.408, viewer_experience=1.000, watch_context=1.000, narrative_techniques=0.988, production=0.100, reception=0.755
Vector-space weights:  anchor=0.123, plot_events=0.042, plot_analysis=0.109, viewer_experience=0.196, watch_context=0.179, narrative_techniques=0.162, production=0.047, reception=0.143

Format bucket: narrative_feature
Consensus countries: ['US_DEFAULT']

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q | Fmt | Th | Cast | Awd |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Mulholland Drive (2001) `1018` | 0.899 | shape | shape, format, themes, specific_award | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.78 | 0.00 | 0.86 |
| 2 | Shutter Island (2010) `11324` | 0.768 | shape | shape, format, themes | 0.91 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.58 | 0.00 | 0.00 |
| 3 | Trance (2013) `68727` | 0.728 | shape | shape, format, themes | 0.85 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.60 | 0.00 | 0.00 |
| 4 | Buster's Mal Heart (2017) `367147` | 0.659 | shape | shape, format, themes | 0.77 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.52 | 0.00 | 0.00 |
| 5 | The I Inside (2004) `11588` | 0.652 | shape | shape, format, themes | 0.72 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.86 | 0.00 | 0.00 |
| 6 | I'm Thinking of Ending Things (2020) `500840` | 0.594 | shape | shape, format, themes, specific_award | 0.68 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.49 | 0.00 | 0.02 |
| 7 | Mentally Apart (2020) `799128` | 0.572 | shape | shape, format, themes | 0.69 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.13 | 0.00 | 0.00 |
| 8 | Eternal Sunshine of the Spotless Mind (2004) `38` | 0.533 | shape | shape, format, themes, specific_award | 0.55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.48 | 0.00 | 0.62 |
| 9 | Extracted (2012) `97605` | 0.532 | shape | shape, format, themes | 0.61 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.37 | 0.00 | 0.00 |
| 10 | Chaos (2000) `39983` | 0.518 | shape | shape, format, themes | 0.80 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.40 | 0.00 | 0.00 |

## Pixar trio (studio + family animation)

Anchors: Toy Story (1995) `862`, Finding Nemo (2003) `12`, Up (2009) `14160`

Active anchor types: standard_shape, studio_lineage, prestige

Lane weights (normalized): shape=0.732, director=0.000, franchise=0.000, studio=0.120, source=0.000, quality=0.080, format=0.054, themes=0.080, cast=0.000, specific_award=0.054
Raw lane weights:           shape=1.095, director=0.000, franchise=0.000, studio=0.120, source=0.000, quality=0.120, format=0.080, themes=0.120, cast=0.000, specific_award=0.080

Vector-space cohesion: anchor=0.593, plot_events=0.172, plot_analysis=0.472, viewer_experience=1.000, watch_context=1.000, narrative_techniques=0.899, production=1.000, reception=0.781
Vector-space weights:  anchor=0.105, plot_events=0.038, plot_analysis=0.105, viewer_experience=0.174, watch_context=0.158, narrative_techniques=0.132, production=0.158, reception=0.130

Format bucket: narrative_feature
Consensus countries: ['US_DEFAULT']

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q | Fmt | Th | Cast | Awd |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Toy Story 3 (2010) `10193` | 1.146 | shape | shape, quality, format, themes, specific_award, studio | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 | 0.61 | 0.00 | 0.60 |
| 2 | Inside Out (2015) `150540` | 0.932 | shape | shape, quality, format, themes, specific_award, studio | 0.77 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 | 0.61 | 0.00 | 0.49 |
| 3 | Monsters, Inc. (2001) `585` | 0.902 | shape | shape, format, themes, specific_award, studio | 0.81 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.82 | 0.00 | 0.63 |
| 4 | Cars (2006) `920` | 0.784 | shape | shape, format, themes, specific_award, studio | 0.73 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.56 | 0.00 | 0.34 |
| 5 | Toy Story 2 (1999) `863` | 0.758 | shape | shape, format, themes, specific_award, studio | 0.65 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.83 | 0.00 | 0.49 |
| 6 | Toy Story 4 (2019) `301528` | 0.733 | shape | shape, format, themes, specific_award, studio | 0.66 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.61 | 0.00 | 0.39 |
| 7 | The Good Dinosaur (2015) `105864` | 0.717 | shape | shape, format, themes, specific_award, studio | 0.62 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.91 | 0.00 | 0.17 |
| 8 | Onward (2020) `508439` | 0.707 | shape | shape, format, themes, specific_award, studio | 0.61 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.95 | 0.00 | 0.17 |
| 9 | Ratatouille (2007) `2062` | 0.613 | shape | shape, quality, format, themes, specific_award, studio | 0.45 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 | 0.77 | 0.00 | 0.65 |
| 10 | A Bug's Life (1998) `9487` | 0.576 | shape | shape, format, themes, specific_award, studio | 0.53 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.78 | 0.00 | 0.34 |

## Ghibli / Miyazaki trio

Anchors: Spirited Away (2001) `129`, My Neighbor Totoro (1988) `8392`, Princess Mononoke (1997) `128`

Active anchor types: standard_shape, studio_lineage

Lane weights (normalized): shape=0.716, director=0.155, franchise=0.000, studio=0.120, source=0.000, quality=0.000, format=0.052, themes=0.077, cast=0.000, specific_award=0.000
Raw lane weights:           shape=1.111, director=0.240, franchise=0.000, studio=0.120, source=0.000, quality=0.000, format=0.080, themes=0.120, cast=0.000, specific_award=0.000

Vector-space cohesion: anchor=0.677, plot_events=0.252, plot_analysis=0.569, viewer_experience=1.000, watch_context=0.945, narrative_techniques=0.888, production=1.000, reception=0.794
Vector-space weights:  anchor=0.113, plot_events=0.047, plot_analysis=0.114, viewer_experience=0.169, watch_context=0.147, narrative_techniques=0.127, production=0.154, reception=0.128

Format bucket: narrative_feature
Consensus countries: ['100']

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q | Fmt | Th | Cast | Awd |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Ponyo (2008) `12429` | 1.194 | shape | shape, director, format, themes, studio | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.83 | 0.00 | 0.00 |
| 2 | Howl's Moving Castle (2004) `4935` | 1.064 | shape | shape, director, format, themes, studio | 0.87 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.67 | 0.00 | 0.00 |
| 3 | Castle in the Sky (1986) `10515` | 0.836 | shape | shape, director, format, themes, studio | 0.60 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.67 | 0.00 | 0.00 |
| 4 | The Tale of The Princess Kaguya (2013) `149871` | 0.806 | shape | shape, format, themes, studio | 0.80 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.53 | 0.00 | 0.00 |
| 5 | The Secret World of Arrietty (2010) `51739` | 0.676 | shape | shape, format, themes, studio | 0.65 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.50 | 0.00 | 0.00 |
| 6 | The Boy and the Heron (2023) `508883` | 0.620 | shape | shape, director, format, themes, studio | 0.46 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.32 | 0.00 | 0.00 |
| 7 | Porco Rosso (1992) `11621` | 0.598 | shape | shape, director, format, themes, studio | 0.43 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.38 | 0.00 | 0.00 |
| 8 | Pokémon 3: The Movie (2000) `10991` | 0.595 | shape | shape, format, themes | 0.62 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.55 | 0.00 | 0.00 |
| 9 | The Wind Rises (2013) `149870` | 0.528 | shape | shape, director, format, themes, studio | 0.36 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.18 | 0.00 | 0.00 |
| 10 | Little Nemo: Adventures in Slumberland (1989) `22611` | 0.510 | shape | shape, format, themes | 0.51 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.64 | 0.00 | 0.00 |

## MCU trio (franchise-shared-universe)

Anchors: Iron Man (2008) `1726`, The Avengers (2012) `24428`, Captain America: Civil War (2016) `271110`

Active anchor types: standard_shape, franchise_dominant, studio_lineage, source_material

Lane weights (normalized): shape=0.605, director=0.000, franchise=0.144, studio=0.120, source=0.048, quality=0.000, format=0.048, themes=0.072, cast=0.036, specific_award=0.048
Raw lane weights:           shape=1.011, director=0.000, franchise=0.240, studio=0.120, source=0.080, quality=0.000, format=0.080, themes=0.120, cast=0.060, specific_award=0.080

Vector-space cohesion: anchor=0.701, plot_events=0.355, plot_analysis=0.583, viewer_experience=1.000, watch_context=0.911, narrative_techniques=0.782, production=0.100, reception=0.898
Vector-space weights:  anchor=0.129, plot_events=0.067, plot_analysis=0.129, viewer_experience=0.188, watch_context=0.159, narrative_techniques=0.127, production=0.045, reception=0.157

Format bucket: narrative_feature
Consensus countries: ['US_DEFAULT']

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q | Fmt | Th | Cast | Awd |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Avengers: Age of Ultron (2015) `99861` | 1.074 | shape | shape, franchise, source, format, themes, cast, studio | 1.00 | 0.00 | 1.00 | 1.00 | 0.35 | 0.00 | 1.00 | 0.53 | 1.00 | 0.00 |
| 2 | Iron Man 2 (2010) `10138` | 1.059 | shape | shape, franchise, source, format, themes, cast, specific_award, studio | 0.91 | 0.00 | 1.00 | 1.00 | 0.35 | 0.00 | 1.00 | 0.96 | 1.00 | 0.28 |
| 3 | Captain America: The Winter Soldier (2014) `100402` | 0.748 | shape | shape, franchise, source, format, themes, cast, specific_award, studio | 0.58 | 0.00 | 1.00 | 1.00 | 0.35 | 0.00 | 1.00 | 0.75 | 0.67 | 0.84 |
| 4 | The Rock (1996) `9802` | 0.493 | shape | shape, format, themes, specific_award | 0.60 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.47 | 0.00 | 0.03 |
| 5 | Iron Man 3 (2013) `68721` | 0.736 | shape | shape, franchise, source, format, themes, cast, specific_award, studio | 0.57 | 0.00 | 1.00 | 1.00 | 0.35 | 0.00 | 1.00 | 0.53 | 1.00 | 0.84 |
| 6 | Guardians of the Galaxy (2014) `118340` | 0.653 | shape | shape, franchise, source, format, themes, specific_award, studio | 0.50 | 0.00 | 1.00 | 1.00 | 0.35 | 0.00 | 1.00 | 0.62 | 0.00 | 0.84 |
| 7 | Avengers: Infinity War (2018) `299536` | 0.576 | shape | shape, franchise, source, format, themes, cast, specific_award, studio | 0.30 | 0.00 | 1.00 | 1.00 | 0.35 | 0.00 | 1.00 | 0.87 | 1.00 | 0.75 |
| 8 | Captain America: Brave New World (2025) `822119` | 0.557 | shape | shape, franchise, source, format, themes, studio | 0.38 | 0.00 | 1.00 | 1.00 | 0.35 | 0.00 | 1.00 | 0.91 | 0.00 | 0.00 |
| 9 | Spider-Man: Far From Home (2019) `429617` | 0.554 | shape | shape, franchise, source, format, themes, specific_award, studio | 0.40 | 0.00 | 1.00 | 1.00 | 0.35 | 0.00 | 1.00 | 0.52 | 0.00 | 0.28 |
| 10 | Captain America: The First Avenger (2011) `1771` | 0.522 | shape | shape, franchise, source, format, themes, cast, studio | 0.29 | 0.00 | 1.00 | 1.00 | 0.35 | 0.00 | 1.00 | 0.91 | 0.67 | 0.00 |

## Stephen King horror (source material)

Anchors: The Shining (1980) `694`, It (2017) `346364`, Misery (1990) `1700`

Active anchor types: standard_shape, source_material

Lane weights (normalized): shape=0.774, director=0.000, franchise=0.000, studio=0.000, source=0.065, quality=0.000, format=0.065, themes=0.097, cast=0.000, specific_award=0.000
Raw lane weights:           shape=0.957, director=0.000, franchise=0.000, studio=0.000, source=0.080, quality=0.000, format=0.080, themes=0.120, cast=0.000, specific_award=0.000

Vector-space cohesion: anchor=0.399, plot_events=0.100, plot_analysis=0.295, viewer_experience=1.000, watch_context=0.954, narrative_techniques=0.875, production=0.100, reception=0.809
Vector-space weights:  anchor=0.098, plot_events=0.034, plot_analysis=0.100, viewer_experience=0.212, watch_context=0.186, narrative_techniques=0.157, production=0.050, reception=0.163

Format bucket: narrative_feature
Consensus countries: ['US_DEFAULT']

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q | Fmt | Th | Cast | Awd |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1408 (2007) `3021` | 0.985 | shape | shape, format, themes | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.58 | 0.00 | 0.00 |
| 2 | Ghost Story (1981) `24634` | 0.882 | shape | shape, source, format, themes | 0.88 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 1.00 | 0.49 | 0.00 | 0.00 |
| 3 | Caveat (2021) `744746` | 0.842 | shape | shape, format, themes | 0.83 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.58 | 0.00 | 0.00 |
| 4 | In Our Blood (2025) `1312228` | 0.835 | shape | shape, format, themes | 0.87 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.21 | 0.00 | 0.00 |
| 5 | Afraid of the Dark (1991) `60158` | 0.810 | shape | shape, format, themes | 0.82 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.42 | 0.00 | 0.00 |
| 6 | The Forbidden Door (2009) `39024` | 0.763 | shape | shape, source, format, themes | 0.77 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 1.00 | 0.21 | 0.00 | 0.00 |
| 7 | Nocturnal Animals (2016) `340666` | 0.743 | shape | shape, source, format, themes | 0.74 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 1.00 | 0.26 | 0.00 | 0.00 |
| 8 | The Woods (2006) `6948` | 0.735 | shape | shape, format, themes | 0.72 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.45 | 0.00 | 0.00 |
| 9 | Halloween III: Season of the Witch (1982) `10676` | 0.732 | shape | shape, format, themes | 0.70 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.58 | 0.00 | 0.00 |
| 10 | Sole Survivor (1984) `29611` | 0.730 | shape | shape, format, themes | 0.73 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.35 | 0.00 | 0.00 |

## Best Picture prestige drama (no shared metadata)

Anchors: The Godfather (1972) `238`, Schindler's List (1993) `424`, 12 Years a Slave (2013) `76203`

Active anchor types: standard_shape, source_material, prestige

Lane weights (normalized): shape=0.649, director=0.000, franchise=0.000, studio=0.000, source=0.060, quality=0.090, format=0.051, themes=0.090, cast=0.000, specific_award=0.060
Raw lane weights:           shape=0.863, director=0.000, franchise=0.000, studio=0.000, source=0.080, quality=0.120, format=0.068, themes=0.120, cast=0.000, specific_award=0.080

Vector-space cohesion: anchor=0.359, plot_events=0.100, plot_analysis=0.100, viewer_experience=1.000, watch_context=0.846, narrative_techniques=0.737, production=0.100, reception=0.587
Vector-space weights:  anchor=0.103, plot_events=0.039, plot_analysis=0.077, viewer_experience=0.238, watch_context=0.190, narrative_techniques=0.153, production=0.057, reception=0.144

Format bucket: documentary
Consensus countries: ['US_DEFAULT']

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q | Fmt | Th | Cast | Awd |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | The Pianist (2002) `423` | 0.968 | shape | shape, source, quality, format, themes, specific_award | 0.96 | 0.00 | 0.00 | 0.00 | 0.23 | 1.00 | 1.00 | 0.69 | 0.00 | 0.64 |
| 2 | The Killing Fields (1984) `625` | 0.802 | shape | shape, source, format, themes, specific_award | 0.87 | 0.00 | 0.00 | 0.00 | 0.17 | 0.00 | 1.00 | 0.50 | 0.00 | 0.95 |
| 3 | Killers of the Flower Moon (2023) `466420` | 0.693 | shape | shape, source, format, themes, specific_award | 0.69 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 1.00 | 0.77 | 0.00 | 0.87 |
| 4 | The Elephant Man (1980) `1955` | 0.558 | shape | shape, source, format, themes, specific_award | 0.54 | 0.00 | 0.00 | 0.00 | 0.17 | 0.00 | 1.00 | 0.56 | 0.00 | 0.74 |
| 5 | The Mission (1986) `11416` | 0.730 | shape | shape, source, themes, specific_award | 0.84 | 0.00 | 0.00 | 0.00 | 0.14 | 0.00 | 0.00 | 0.64 | 0.00 | 0.85 |
| 6 | The Shop on Main Street (1965) `25905` | 0.724 | shape | shape, themes, specific_award | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.00 | 0.05 |
| 7 | Citizen Kane (1941) `15` | 0.686 | shape | shape, quality, themes, specific_award | 0.69 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.56 | 0.00 | 0.62 |
| 8 | Katyn (2007) `13614` | 0.663 | shape | shape, source, themes, specific_award | 0.87 | 0.00 | 0.00 | 0.00 | 0.14 | 0.00 | 0.00 | 0.35 | 0.00 | 0.01 |
| 9 | Holy Cross (2003) `438345` | 0.633 | shape | shape, themes, specific_award | 0.88 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.04 |
| 10 | The Grapes of Wrath (1940) `596` | 0.627 | shape | shape, source, quality, themes, specific_award | 0.63 | 0.00 | 0.00 | 0.00 | 0.20 | 1.00 | 0.00 | 0.28 | 0.00 | 0.51 |

## Cult-garbage trio

Anchors: Sharknado (2013) `205321`, The Room (2003) `17473`, Birdemic: Shock and Terror (2010) `40016`

Active anchor types: standard_shape, cult_garbage

Lane weights (normalized): shape=0.738, director=0.000, franchise=0.000, studio=0.000, source=0.000, quality=0.098, format=0.065, themes=0.098, cast=0.000, specific_award=0.000
Raw lane weights:           shape=0.902, director=0.000, franchise=0.000, studio=0.000, source=0.000, quality=0.120, format=0.080, themes=0.120, cast=0.000, specific_award=0.000

Vector-space cohesion: anchor=0.506, plot_events=0.100, plot_analysis=0.204, viewer_experience=1.000, watch_context=1.000, narrative_techniques=0.669, production=0.100, reception=0.854
Vector-space weights:  anchor=0.117, plot_events=0.035, plot_analysis=0.087, viewer_experience=0.215, watch_context=0.196, narrative_techniques=0.127, production=0.051, reception=0.173

Format bucket: narrative_feature
Consensus countries: ['US_DEFAULT']

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q | Fmt | Th | Cast | Awd |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Troll 2 (1990) `26914` | 1.043 | shape | shape, quality, format, themes | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.48 | 0.00 | 0.00 |
| 2 | Space Mutiny (1988) `32148` | 0.848 | shape | shape, quality, format, themes | 0.83 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 | 0.29 | 0.00 | 0.00 |
| 3 | Plan 9 from Outer Space (1957) `10513` | 0.787 | shape | shape, format, themes | 0.82 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.48 | 0.00 | 0.00 |
| 4 | Birdemic 2: The Resurrection (2013) `188489` | 0.714 | shape | shape, quality, format, themes | 0.61 | 0.00 | 0.00 | 0.00 | 0.00 | 0.59 | 1.00 | 0.75 | 0.00 | 0.00 |
| 5 | Sharknado 4: The 4th Awakens (2016) `390989` | 0.696 | shape | shape, format, themes | 0.64 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| 6 | Mega Shark vs. Giant Octopus (2009) `17911` | 0.643 | shape | shape, quality, format, themes | 0.46 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.80 | 0.00 | 0.00 |
| 7 | Mayday (2019) `626576` | 0.610 | shape | shape, format, themes | 0.62 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.36 | 0.00 | 0.00 |
| 8 | Sharknado 2: The Second One (2014) `248504` | 0.600 | shape | shape, format, themes | 0.52 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| 9 | Hard Rock Zombies (1985) `28128` | 0.567 | shape | shape, format, themes | 0.59 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.17 | 0.00 | 0.00 |
| 10 | Mega Shark vs. Crocosaurus (2010) `52454` | 0.538 | shape | shape, format, themes | 0.47 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.75 | 0.00 | 0.00 |

## Tarantino trio (director + studio)

Anchors: Pulp Fiction (1994) `680`, Reservoir Dogs (1992) `500`, Kill Bill: Vol. 1 (2003) `24`

Active anchor types: standard_shape, studio_lineage

Lane weights (normalized): shape=0.635, director=0.157, franchise=0.000, studio=0.101, source=0.000, quality=0.000, format=0.052, themes=0.079, cast=0.033, specific_award=0.044
Raw lane weights:           shape=0.970, director=0.240, franchise=0.000, studio=0.101, source=0.000, quality=0.000, format=0.080, themes=0.120, cast=0.051, specific_award=0.068

Vector-space cohesion: anchor=0.546, plot_events=0.393, plot_analysis=0.337, viewer_experience=1.000, watch_context=0.702, narrative_techniques=0.696, production=0.323, reception=0.748
Vector-space weights:  anchor=0.117, plot_events=0.078, plot_analysis=0.103, viewer_experience=0.205, watch_context=0.141, narrative_techniques=0.125, production=0.083, reception=0.148

Format bucket: narrative_feature
Consensus countries: ['US_DEFAULT']

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q | Fmt | Th | Cast | Awd |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | True Romance (1993) `319` | 0.840 | shape | shape, format, themes, studio | 0.98 | 0.00 | 0.00 | 0.67 | 0.00 | 0.00 | 1.00 | 0.49 | 0.00 | 0.00 |
| 2 | Kill Bill: Vol. 2 (2004) `393` | 0.808 | shape | shape, director, format, themes, cast, specific_award, studio | 0.66 | 1.00 | 0.00 | 0.67 | 0.00 | 0.00 | 1.00 | 0.38 | 0.33 | 0.41 |
| 3 | Lucky Number Slevin (2006) `186` | 0.614 | shape | shape, format, themes, cast | 0.71 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.58 | 0.33 | 0.00 |
| 4 | Caliber 9 (1972) `40022` | 0.609 | shape | shape, format, themes | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.38 | 0.00 | 0.00 |
| 5 | Truth or Consequences, N.M. (1997) `31017` | 0.606 | shape | shape, format, themes | 0.72 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.53 | 0.00 | 0.00 |
| 6 | Sin City (2005) `187` | 0.583 | shape | shape, director, format, themes, specific_award | 0.45 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.42 | 0.00 | 0.03 |
| 7 | Inglourious Basterds (2009) `16869` | 0.582 | shape | shape, director, format, themes, specific_award | 0.43 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.17 | 0.00 | 0.68 |
| 8 | Man on Fire (2004) `9509` | 0.582 | shape | shape, format, themes, specific_award | 0.68 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.58 | 0.00 | 0.03 |
| 9 | Tazza: The High Rollers (2006) `38015` | 0.572 | shape | shape, format, themes | 0.95 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.18 | 0.00 | 0.00 |
| 10 | Bullets, Blood & a Fistful of Ca$h (2006) `44817` | 0.523 | shape | shape, format, themes | 0.63 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.31 | 0.00 | 0.00 |

## Heist trio (vector-only, no metadata cohesion)

Anchors: Ocean's Eleven (2001) `161`, Heat (1995) `949`, The Italian Job (2003) `9654`

Active anchor types: standard_shape, source_material

Lane weights (normalized): shape=0.791, director=0.000, franchise=0.000, studio=0.000, source=0.055, quality=0.000, format=0.055, themes=0.098, cast=0.000, specific_award=0.000
Raw lane weights:           shape=0.967, director=0.000, franchise=0.000, studio=0.000, source=0.068, quality=0.000, format=0.068, themes=0.120, cast=0.000, specific_award=0.000

Vector-space cohesion: anchor=0.555, plot_events=0.100, plot_analysis=0.807, viewer_experience=1.000, watch_context=0.847, narrative_techniques=1.000, production=0.100, reception=0.581
Vector-space weights:  anchor=0.114, plot_events=0.032, plot_analysis=0.169, viewer_experience=0.197, watch_context=0.157, narrative_techniques=0.165, production=0.047, reception=0.118

Format bucket: narrative_feature
Consensus countries: ['US_DEFAULT']

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q | Fmt | Th | Cast | Awd |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Ocean's Thirteen (2007) `298` | 1.016 | shape | shape, format, themes | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.79 | 0.00 | 0.00 |
| 2 | Foolproof (2003) `14527` | 0.992 | shape | shape, format, themes | 0.95 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.95 | 0.00 | 0.00 |
| 3 | The Italian Job (1969) `10536` | 0.949 | shape | shape, format, themes | 0.92 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.83 | 0.00 | 0.00 |
| 4 | Fast Five (2011) `51497` | 0.924 | shape | shape, format, themes | 0.91 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.66 | 0.00 | 0.00 |
| 5 | Lift (2024) `955916` | 0.883 | shape | shape, format, themes | 0.85 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.78 | 0.00 | 0.00 |
| 6 | Den of Thieves (2018) `449443` | 0.815 | shape | shape, format, themes | 0.78 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.67 | 0.00 | 0.00 |
| 7 | Takers (2010) `22907` | 0.814 | shape | shape, format, themes | 0.76 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.87 | 0.00 | 0.00 |
| 8 | The Rise (2012) `128241` | 0.766 | shape | shape, format, themes | 0.75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 | 0.00 | 0.00 |
| 9 | The Outfit (1973) `26762` | 0.750 | shape | shape, format, themes | 0.73 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.47 | 0.00 | 0.00 |
| 10 | The Heist (1989) `54535` | 0.719 | shape | shape, format, themes | 0.72 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.26 | 0.00 | 0.00 |

## LotR trilogy (everything cohesive)

Anchors: The Lord of the Rings: The Fellowship of the Ring (2001) `120`, The Lord of the Rings: The Two Towers (2002) `121`, The Lord of the Rings: The Return of the King (2003) `122`

Active anchor types: standard_shape, franchise_dominant, studio_lineage, source_material, prestige

Lane weights (normalized): shape=0.541, director=0.108, franchise=0.108, studio=0.120, source=0.036, quality=0.054, format=0.036, themes=0.054, cast=0.027, specific_award=0.036
Raw lane weights:           shape=1.200, director=0.240, franchise=0.240, studio=0.120, source=0.080, quality=0.120, format=0.080, themes=0.120, cast=0.060, specific_award=0.080

Vector-space cohesion: anchor=0.982, plot_events=0.872, plot_analysis=0.975, viewer_experience=1.000, watch_context=1.000, narrative_techniques=0.929, production=1.000, reception=1.000
Vector-space weights:  anchor=0.126, plot_events=0.104, plot_analysis=0.137, viewer_experience=0.140, watch_context=0.128, narrative_techniques=0.110, production=0.128, reception=0.128

Format bucket: narrative_feature
Consensus countries: ['US_DEFAULT']

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q | Fmt | Th | Cast | Awd |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | The Hobbit: An Unexpected Journey (2012) `49051` | 0.905 | shape | shape, director, franchise, source, format, themes, specific_award, studio | 0.82 | 1.00 | 1.00 | 1.00 | 0.20 | 0.00 | 1.00 | 0.64 | 0.00 | 0.31 |
| 2 | The Hobbit: The Battle of the Five Armies (2014) `122917` | 0.872 | shape | shape, director, franchise, source, format, themes, specific_award, studio | 0.77 | 1.00 | 1.00 | 1.00 | 0.20 | 0.00 | 1.00 | 0.71 | 0.00 | 0.17 |
| 3 | The Lord of the Rings (1978) `123` | 0.802 | shape | shape, franchise, source, format, themes, specific_award | 1.00 | 0.00 | 1.00 | 0.00 | 0.20 | 0.00 | 1.00 | 0.64 | 0.00 | 0.08 |
| 4 | The Hobbit: The Desolation of Smaug (2013) `57158` | 0.792 | shape | shape, director, source, format, themes, specific_award, studio | 0.85 | 1.00 | 0.00 | 1.00 | 0.20 | 0.00 | 1.00 | 0.62 | 0.00 | 0.29 |
| 5 | The Dark Knight Rises (2012) `49026` | 0.490 | shape | shape, format, themes, specific_award | 0.71 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.29 | 0.00 | 0.25 |
| 6 | The Return of the King (1980) `1361` | 0.457 | shape | shape, franchise, source, format, themes | 0.46 | 0.00 | 1.00 | 0.00 | 0.20 | 0.00 | 1.00 | 0.32 | 0.00 | 0.00 |
| 7 | Harry Potter and the Deathly Hallows: Part 2 (2011) `12445` | 0.443 | shape | shape, source, format, themes, specific_award | 0.57 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 1.00 | 0.76 | 0.00 | 0.35 |
| 8 | Gladiator (2000) `98` | 0.443 | shape | shape, format, themes, specific_award | 0.57 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.46 | 0.00 | 0.90 |
| 9 | Star Wars (1977) `11` | 0.429 | shape | shape, quality, format, themes, specific_award | 0.42 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.86 | 0.00 | 0.68 |
| 10 | King Kong (2005) `254` | 0.425 | shape | shape, director, format, themes, specific_award | 0.38 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.38 | 0.00 | 0.42 |

## Chaotic mixed bag (intentionally incohesive)

Anchors: Toy Story (1995) `862`, The Godfather (1972) `238`, Sharknado (2013) `205321`

Active anchor types: standard_shape, prestige

Lane weights (normalized): shape=0.639, director=0.000, franchise=0.000, studio=0.000, source=0.000, quality=0.104, format=0.082, themes=0.104, cast=0.000, specific_award=0.070
Raw lane weights:           shape=0.621, director=0.000, franchise=0.000, studio=0.000, source=0.000, quality=0.101, format=0.080, themes=0.101, cast=0.000, specific_award=0.068

Vector-space cohesion: anchor=0.100, plot_events=0.100, plot_analysis=0.100, viewer_experience=0.963, watch_context=0.206, narrative_techniques=0.531, production=0.100, reception=0.100
Vector-space weights:  anchor=0.080, plot_events=0.055, plot_analysis=0.109, viewer_experience=0.327, watch_context=0.106, narrative_techniques=0.163, production=0.080, reception=0.080

Format bucket: narrative_feature
Consensus countries: ['US_DEFAULT']

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q | Fmt | Th | Cast | Awd |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Toy Story 3 (2010) `10193` | 0.860 | shape | shape, quality, format, themes, specific_award | 0.77 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 | 0.90 | 0.00 | 0.64 |
| 2 | Star Wars (1977) `11` | 0.731 | shape | shape, quality, format, themes, specific_award | 0.61 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 | 0.50 | 0.00 | 1.00 |
| 3 | Life or Something Like It (2002) `16643` | 0.723 | shape | shape, format, themes | 0.87 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.20 | 0.00 | 0.00 |
| 4 | Mission to Mars (2000) `2067` | 0.622 | shape | shape, format, themes | 0.70 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.34 | 0.00 | 0.00 |
| 5 | A Bronx Tale (1993) `1607` | 0.741 | shape | shape, themes, specific_award | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.13 | 0.00 | 0.30 |
| 6 | Star Trek III: The Search for Spock (1984) `157` | 0.604 | shape | shape, format, themes | 0.66 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.43 | 0.00 | 0.00 |
| 7 | Sharknado 4: The 4th Awakens (2016) `390989` | 0.576 | shape | shape, format, themes | 0.54 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.90 | 0.00 | 0.00 |
| 8 | Phantom of the Opera (1943) `15855` | 0.551 | shape | shape, format, themes, specific_award | 0.58 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.16 | 0.00 | 0.39 |
| 9 | The Incredibles (2004) `9806` | 0.496 | shape | shape, quality, format, themes, specific_award | 0.26 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 | 0.79 | 0.00 | 0.73 |
| 10 | Sword in the Desert (1949) `293258` | 0.479 | shape | shape, format, themes | 0.54 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.07 | 0.00 | 0.00 |

## War film trio (prestige + vector)

Anchors: Saving Private Ryan (1998) `857`, 1917 (2019) `530915`, Apocalypse Now (1979) `28`

Active anchor types: standard_shape, prestige

Lane weights (normalized): shape=0.720, director=0.000, franchise=0.000, studio=0.000, source=0.000, quality=0.075, format=0.059, themes=0.088, cast=0.000, specific_award=0.059
Raw lane weights:           shape=0.979, director=0.000, franchise=0.000, studio=0.000, source=0.000, quality=0.101, format=0.080, themes=0.120, cast=0.000, specific_award=0.080

Vector-space cohesion: anchor=0.519, plot_events=0.123, plot_analysis=0.614, viewer_experience=1.000, watch_context=0.929, narrative_techniques=0.764, production=0.100, reception=0.927
Vector-space weights:  anchor=0.109, plot_events=0.036, plot_analysis=0.140, viewer_experience=0.198, watch_context=0.170, narrative_techniques=0.131, production=0.047, reception=0.170

Format bucket: narrative_feature
Consensus countries: ['US_DEFAULT']

| # | Result | Score | Dominant | Lanes | Shape | Dir | Fr | St | Src | Q | Fmt | Th | Cast | Awd |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | All Quiet on the Western Front (2022) `49046` | 0.764 | shape | shape, format, themes, specific_award | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.85 | 0.00 | 0.78 |
| 2 | Dunkirk (2017) `374720` | 0.707 | shape | shape, quality, format, themes, specific_award | 0.59 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 | 0.69 | 0.00 | 0.80 |
| 3 | All Quiet on the Western Front (1930) `143` | 0.601 | shape | shape, quality, format, themes, specific_award | 0.49 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 | 0.73 | 0.00 | 0.33 |
| 4 | The Thin Red Line (1964) `188608` | 0.553 | shape | shape, format, themes | 0.58 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.26 | 0.00 | 0.00 |
| 5 | Platoon (1986) `792` | 0.661 | shape | shape, quality, themes, specific_award | 0.65 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 | 0.54 | 0.00 | 0.66 |
| 6 | The Thin Red Line (1998) `8741` | 0.541 | shape | shape, format, themes, specific_award | 0.46 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.77 | 0.00 | 0.58 |
| 7 | Fury (2014) `228150` | 0.530 | shape | shape, format, themes, specific_award | 0.51 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.54 | 0.00 | 0.15 |
| 8 | Kajaki (2014) `306650` | 0.530 | shape | shape, format, themes, specific_award | 0.55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.31 | 0.00 | 0.01 |
| 9 | Saints and Soldiers (2003) `10105` | 0.521 | shape | shape, format, themes, specific_award | 0.52 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.38 | 0.00 | 0.12 |
| 10 | Journey's End (2017) `438259` | 0.500 | shape | shape, format, themes | 0.52 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.28 | 0.00 | 0.00 |
