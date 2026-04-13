"""OverallKeyword enum for IMDB's curated 225-term genre/sub-genre taxonomy."""

from enum import Enum

from implementation.misc.helpers import normalize_string


class OverallKeyword(Enum):
    """Enum of IMDB overall_keywords with stable numeric IDs.

    Each value represents a curated genre or sub-genre tag from IMDB's
    overall_keywords system (225 terms total). These are distinct from
    the free-form plot_keywords — overall_keywords is a compact,
    editorially managed taxonomy with 100% movie coverage.

    Each member also carries a short LLM-facing definition that clarifies
    what the keyword means and when it should apply to a movie.
    """

    keyword_id: int
    value: str
    definition: str

    def __new__(
        cls,
        keyword_id: int,
        value: str,
        definition: str,
    ) -> "OverallKeyword":
        """Create an OverallKeyword enum member with an ID, label, and definition."""
        obj = object.__new__(cls)
        obj._value_ = value
        obj.keyword_id = keyword_id
        obj.definition = definition
        return obj

    ACTION = (1, "Action", "A fast-paced movie driven mainly by physical conflict, chases, fights, and dangerous feats.")  # 16,834 movies
    ACTION_EPIC = (2, "Action Epic", "An action movie with large-scale spectacle and a sweeping, high-stakes conflict of major consequence.")  # 386 movies
    ADULT_ANIMATION = (3, "Adult Animation", "An animated movie aimed mainly at adults, with mature themes, humor, violence, or sexual content.")  # 751 movies
    ADVENTURE = (4, "Adventure", "A movie centered on an exciting journey, quest, or expedition full of risk, obstacles, and discovery.")  # 10,574 movies
    ADVENTURE_EPIC = (5, "Adventure Epic", "An adventure movie with grand scope, heroic journeys, and major challenges across sweeping or mythic settings.")  # 334 movies
    ALIEN_INVASION = (6, "Alien Invasion", "A movie about extraterrestrials arriving on Earth as an invading or occupying force humanity must confront.")  # 374 movies
    ANIMAL_ADVENTURE = (7, "Animal Adventure", "An adventure movie where animals are central protagonists and their journey, survival, or mission drives the story.")  # 486 movies
    ANIMATION = (8, "Animation", "A movie told through animated moving images rather than primarily live-action footage.")  # 6,010 movies
    ANIME = (9, "Anime", "An animated movie in the Japanese anime style, typically made in Japan or strongly modeled on that tradition.")  # 929 movies
    ARABIC = (10, "Arabic", "A movie whose primary spoken language is Arabic.")  # 403 movies
    ARTIFICIAL_INTELLIGENCE = (11, "Artificial Intelligence", "A movie centered on intelligent machines or software and their effects on people, society, or the future.")  # 147 movies
    B_ACTION = (12, "B-Action", "A low-budget action movie with modest production values and a pulpy or campy feel.")  # 440 movies
    B_HORROR = (13, "B-Horror", "A low-budget horror movie with modest production values and a pulpy, campy, or exploitative feel.")  # 983 movies
    BASEBALL = (14, "Baseball", "A sports movie where baseball is central to the plot and the characters' struggles or goals.")  # 85 movies
    BASKETBALL = (15, "Basketball", "A sports movie where basketball is central to the plot and the characters' struggles or goals.")  # 117 movies
    BENGALI = (16, "Bengali", "A movie whose primary spoken language is Bengali.")  # 293 movies
    BIOGRAPHY = (17, "Biography", "A movie that dramatizes the life of a real person, usually focusing on their major experiences, struggles, and impact.")  # 4,570 movies
    BODY_HORROR = (18, "Body Horror", "A horror movie where bodily transformation, mutilation, mutation, or physical degradation is the main source of fear.")  # 907 movies
    BODY_SWAP_COMEDY = (19, "Body Swap Comedy", "A comedy where characters swap bodies or consciousnesses, creating humor from role reversals and life in someone else's body.")  # 69 movies
    BOXING = (20, "Boxing", "A sports movie where boxing is central to the plot and the characters' struggles or goals.")  # 134 movies
    BUDDY_COMEDY = (21, "Buddy Comedy", "A comedy centered on two contrasting companions whose banter, misadventures, and growing friendship drive the story.")  # 612 movies
    BUDDY_COP = (22, "Buddy Cop", "A crime story about two mismatched law-enforcement partners whose contrasting styles drive the case and the humor or action.")  # 106 movies
    BUMBLING_DETECTIVE = (23, "Bumbling Detective", "A mystery-comedy where the investigator is notably clumsy or inept and the humor comes from their blunders.")  # 65 movies
    BUSINESS_REALITY_TV = (24, "Business Reality TV", "A reality-style title focused on business, entrepreneurship, or workplace competition and professional decision-making.")  # 1 movie
    CANTONESE = (25, "Cantonese", "A movie whose primary spoken language is Cantonese.")  # 970 movies
    CAPER = (26, "Caper", "A movie centered on planning and pulling off a clever, often playful crime such as a heist or con.")  # 378 movies
    CAR_ACTION = (27, "Car Action", "An action movie where high-speed chases, races, crashes, and vehicle stunts are central to the story.")  # 137 movies
    CLASSIC_MUSICAL = (28, "Classic Musical", "A Golden Age-style musical built around lavish songs, dances, and classic Hollywood showmanship.")  # 121 movies
    CLASSICAL_WESTERN = (29, "Classical Western", "A traditional, romanticized Old West story with cowboys, frontier justice, and rugged landscapes.")  # 400 movies
    COMEDY = (30, "Comedy", "A movie whose primary purpose is to amuse through humor, wit, and comic situations.")  # 33,054 movies
    COMING_OF_AGE = (31, "Coming-of-Age", "A movie that follows a young person's emotional growth and transition toward adulthood.")  # 1,252 movies
    COMPUTER_ANIMATION = (32, "Computer Animation", "An animated movie whose imagery is created digitally with computers rather than primarily by hand or stop-motion.")  # 792 movies
    CONCERT = (33, "Concert", "A movie that primarily presents a live musical performance.")  # 86 movies
    CONSPIRACY_THRILLER = (34, "Conspiracy Thriller", "A thriller where suspense comes from uncovering a hidden plot run by powerful people or organizations.")  # 384 movies
    CONTEMPORARY_WESTERN = (35, "Contemporary Western", "A movie that uses classic Western themes and archetypes in a modern-day setting.")  # 102 movies
    COOKING_COMPETITION = (36, "Cooking Competition", "A movie centered on chefs or cooks competing in judged culinary challenges or cook-offs.")  # 1 movie
    COP_DRAMA = (37, "Cop Drama", "A movie where police officers are the main characters and the story focuses on their cases, work, and personal pressures.")  # 187 movies
    COSTUME_DRAMA = (38, "Costume Drama", "A period drama that strongly emphasizes historical clothing, customs, and social world.")  # 326 movies
    COZY_MYSTERY = (39, "Cozy Mystery", "A light, puzzle-driven mystery with little graphic violence, often set in a close-knit community.")  # 90 movies
    CRIME = (40, "Crime", "A movie revolving around criminal acts, their investigation, or the pursuit of justice.")  # 15,332 movies
    CRIME_DOCUMENTARY = (41, "Crime Documentary", "A nonfiction movie that examines real criminal cases, investigations, trials, or perpetrators.")  # 243 movies
    CYBER_THRILLER = (42, "Cyber Thriller", "A thriller where suspense is driven by hacking, cybercrime, digital systems, or networked technology.")  # 99 movies
    CYBERPUNK = (43, "Cyberpunk", "A movie set in a gritty high-tech dystopia where advanced technology collides with social decay and corporate power.")  # 173 movies
    DANISH = (44, "Danish", "A movie whose primary spoken language is Danish.")  # 353 movies
    DARK_COMEDY = (45, "Dark Comedy", "A comedy that treats death, crime, cruelty, or other grim subjects with humor.")  # 3,447 movies
    DARK_FANTASY = (46, "Dark Fantasy", "A fantasy movie that combines magic or mythic elements with horror, menace, or a macabre atmosphere.")  # 488 movies
    DARK_ROMANCE = (47, "Dark Romance", "A romance shaped by obsession, danger, or troubling emotional darkness.")  # 233 movies
    DESERT_ADVENTURE = (48, "Desert Adventure", "An adventure that unfolds mainly in harsh, arid desert environments.")  # 136 movies
    DINOSAUR_ADVENTURE = (49, "Dinosaur Adventure", "An adventure movie where dinosaurs are central to the story, whether in prehistoric or modern rediscovery settings.")  # 145 movies
    DISASTER = (50, "Disaster", "A movie whose main plot centers on surviving or responding to a catastrophic natural or human-made event.")  # 247 movies
    DOCUDRAMA = (51, "Docudrama", "A dramatized retelling of real events that uses reenacted scenes while staying rooted in actual history or real-life situations.")  # 703 movies
    DOCUMENTARY = (52, "Documentary", "A nonfiction film built around real people, events, or issues and intended to inform, examine, or record reality.")  # 7,909 movies
    DRAMA = (53, "Drama", "A serious, emotionally grounded story driven by human conflict, relationships, and character development.")  # 56,798 movies
    DRUG_CRIME = (54, "Drug Crime", "A crime movie centered on illegal drugs, trafficking, addiction, or drug enforcement.")  # 115 movies
    DUTCH = (55, "Dutch", "A movie whose primary spoken language is Dutch.")  # 397 movies
    DYSTOPIAN_SCI_FI = (56, "Dystopian Sci-Fi", "A science-fiction movie set in a damaged or oppressive future society shaped by major social, political, or environmental breakdown.")  # 371 movies
    EPIC = (57, "Epic", "A large-scale drama with sweeping scope, major historical or societal stakes, and a sense of grandeur.")  # 586 movies
    EROTIC_THRILLER = (58, "Erotic Thriller", "A thriller that combines suspense or crime with sexual desire, seduction, and adult relationships.")  # 363 movies
    EXTREME_SPORT = (59, "Extreme Sport", "A movie centered on high-risk, adrenaline-driven sports or outdoor feats requiring daring physical skill.")  # 64 movies
    FAIRY_TALE = (60, "Fairy Tale", "A wonder-filled story built around magic, archetypal characters, and an enchanted moral struggle.")  # 236 movies
    FAITH_AND_SPIRITUALITY_DOCUMENTARY = (61, "Faith & Spirituality Documentary", "A nonfiction film about religious belief, spiritual practice, or the search for existential meaning.")  # 86 movies
    FAMILY = (62, "Family", "A movie made to be appropriate and enjoyable for both children and adults watching together.")  # 9,095 movies
    FANTASY = (63, "Fantasy", "A movie set in a world shaped by magic, mythical beings, supernatural powers, or other impossible elements.")  # 7,661 movies
    FANTASY_EPIC = (64, "Fantasy Epic", "A large-scale fantasy adventure set in an expansive imagined world with heroic quests and major conflicts.")  # 199 movies
    FARCE = (65, "Farce", "A comedy built on absurd exaggeration, mistaken identities, physical humor, and chaotic improbable situations.")  # 626 movies
    FEEL_GOOD_ROMANCE = (66, "Feel-Good Romance", "An uplifting romance designed to leave the viewer happy and emotionally satisfied.")  # 454 movies
    FILIPINO = (67, "Filipino", "A movie whose primary spoken language is Filipino.")  # 282 movies
    FILM_NOIR = (68, "Film Noir", "A dark, stylish crime drama marked by moral ambiguity, cynical characters, shadowy visuals, and a fatalistic mood.")  # 926 movies
    FINANCIAL_DRAMA = (69, "Financial Drama", "A drama where money, markets, business, or economic crisis drives the central conflict.")  # 61 movies
    FINNISH = (70, "Finnish", "A movie whose primary spoken language is Finnish.")  # 179 movies
    FOLK_HORROR = (71, "Folk Horror", "A horror movie that draws fear from folklore, pagan or rural traditions, and uncanny threats tied to isolated communities or landscapes.")  # 969 movies
    FOOD_DOCUMENTARY = (72, "Food Documentary", "A nonfiction film about food, cooking, agriculture, or the food industry, often linked to culture, health, or society.")  # 50 movies
    FOOTBALL = (73, "Football", "A sports movie centered on American football and the lives, competition, or struggles of players, coaches, or teams.")  # 75 movies
    FOUND_FOOTAGE_HORROR = (74, "Found Footage Horror", "A horror movie presented as supposedly recovered recordings to make the events feel raw and real.")  # 309 movies
    FRENCH = (75, "French", "A movie whose primary spoken language is French.")  # 4,440 movies
    GAME_SHOW = (76, "Game Show", "A title that uses a game-show format in which contestants compete in challenges, quizzes, or contests for prizes.")  # 1 movie
    GANGSTER = (77, "Gangster", "A crime movie centered on gangsters or organized-crime figures, often charting an antihero's rise and fall.")  # 383 movies
    GERMAN = (78, "German", "A movie whose primary spoken language is German.")  # 1,801 movies
    GIALLO = (79, "Giallo", "A movie in the Italian giallo tradition, blending mystery, horror, and psychological-thriller elements with stylized violence.")  # 117 movies
    GLOBETROTTING_ADVENTURE = (80, "Globetrotting Adventure", "An adventure where the characters travel across multiple countries or far-flung locations on a high-stakes journey.")  # 159 movies
    GREEK = (81, "Greek", "A movie whose primary spoken language is Greek.")  # 213 movies
    GUN_FU = (82, "Gun Fu", "An action movie with highly choreographed close-quarters gunplay that blends firearms with martial-arts-style movement.")  # 146 movies
    HAND_DRAWN_ANIMATION = (83, "Hand-Drawn Animation", "An animated movie created by drawing each frame by hand in a traditional animation style.")  # 1,591 movies
    HARD_BOILED_DETECTIVE = (84, "Hard-boiled Detective", "A crime story about a tough, cynical investigator navigating crime, corruption, and moral ambiguity.")  # 63 movies
    HEIST = (85, "Heist", "A crime movie centered on planning, carrying out, or escaping the fallout from a major theft or robbery.")  # 235 movies
    HIGH_CONCEPT_COMEDY = (86, "High-Concept Comedy", "A comedy built around a distinctive, easily pitchable premise, often framed as a clear 'what if?' scenario.")  # 261 movies
    HINDI = (87, "Hindi", "A movie whose primary spoken language is Hindi.")  # 3,044 movies
    HISTORICAL_EPIC = (88, "Historical Epic", "A large-scale drama about major historical events, eras, or figures, told with epic scope and spectacle.")  # 137 movies
    HISTORY = (89, "History", "A movie that dramatizes or examines real historical periods, events, societies, or figures.")  # 4,200 movies
    HISTORY_DOCUMENTARY = (90, "History Documentary", "A nonfiction film that explores historical events, people, periods, or cultural developments to explain the past.")  # 81 movies
    HOLIDAY = (91, "Holiday", "A movie centered on a holiday celebration or season, drawing on its traditions, atmosphere, and emotions.")  # 492 movies
    HOLIDAY_ANIMATION = (92, "Holiday Animation", "An animated movie centered on a holiday celebration and its festive traditions or spirit.")  # 121 movies
    HOLIDAY_COMEDY = (93, "Holiday Comedy", "A comedy built around a holiday celebration, using festive settings for humor and seasonal warmth.")  # 190 movies
    HOLIDAY_FAMILY = (94, "Holiday Family", "A family-friendly holiday movie focused on festive traditions, family bonds, and seasonal warmth.")  # 139 movies
    HOLIDAY_ROMANCE = (95, "Holiday Romance", "A romance set during a holiday season, where festive circumstances help drive the love story.")  # 402 movies
    HORROR = (96, "Horror", "A movie intended to evoke fear, dread, or repulsion through frightening, macabre, or disturbing situations.")  # 15,448 movies
    ISEKAI = (97, "Isekai", "An anime story where characters are transported or reborn into another world and must navigate life there.")  # 24 movies
    ITALIAN = (98, "Italian", "A movie whose primary spoken language is Italian.")  # 2,530 movies
    IYASHIKEI = (99, "Iyashikei", "An anime focused on calm everyday life and a soothing atmosphere meant to feel healing, relaxing, and comforting.")  # 15 movies
    JAPANESE = (100, "Japanese", "A movie whose primary spoken language is Japanese.")  # 2,879 movies
    JOSEI = (101, "Josei", "An anime aimed at adult women, often focused on mature relationships, emotions, or everyday life.")  # 9 movies
    JUKEBOX_MUSICAL = (102, "Jukebox Musical", "A musical that tells its story mainly with pre-existing popular songs rather than an original score.")  # 72 movies
    JUNGLE_ADVENTURE = (103, "Jungle Adventure", "An adventure set mainly in a dense, dangerous jungle and centered on exploration, survival, or wildlife threats.")  # 333 movies
    KAIJU = (104, "Kaiju", "A movie where giant monsters are the main spectacle, usually attacking cities or battling other giant creatures.")  # 170 movies
    KANNADA = (105, "Kannada", "A movie whose primary spoken language is Kannada.")  # 233 movies
    KOREAN = (106, "Korean", "A movie whose primary spoken language is Korean.")  # 1,061 movies
    KUNG_FU = (107, "Kung Fu", "A movie centered on Chinese martial arts and emphasizing hand-to-hand fight choreography.")  # 140 movies
    LEGAL_DRAMA = (108, "Legal Drama", "A drama where lawyers, courts, or legal cases are central to the story.")  # 222 movies
    LEGAL_THRILLER = (109, "Legal Thriller", "A thriller where legal cases or lawyers drive suspense built around danger, twists, or hidden evidence.")  # 83 movies
    MALAYALAM = (110, "Malayalam", "A movie whose primary spoken language is Malayalam.")  # 878 movies
    MANDARIN = (111, "Mandarin", "A movie whose primary spoken language is Mandarin Chinese.")  # 1,378 movies
    MARATHI = (112, "Marathi", "A movie whose primary spoken language is Marathi.")  # 158 movies
    MARTIAL_ARTS = (113, "Martial Arts", "A movie where martial-arts combat and choreography are central to the action and story.")  # 724 movies
    MECHA = (114, "Mecha", "A movie where giant robots or mechanical suits are central to the action, usually in science-fiction battles.")  # 62 movies
    MEDICAL_DRAMA = (115, "Medical Drama", "A drama where doctors, patients, or medical settings are central to the story.")  # 117 movies
    MILITARY_DOCUMENTARY = (116, "Military Documentary", "A nonfiction film about armed forces, warfare, military operations, or service members.")  # 63 movies
    MOCKUMENTARY = (117, "Mockumentary", "A fictional movie presented in documentary style, usually for satire or comedy.")  # 80 movies
    MONSTER_HORROR = (118, "Monster Horror", "A horror movie where monsters are the main source of fear and the story centers on confronting or surviving them.")  # 475 movies
    MOTORSPORT = (119, "Motorsport", "A movie where motor racing or competitive driving is central to the story.")  # 69 movies
    MOUNTAIN_ADVENTURE = (120, "Mountain Adventure", "An adventure where characters face danger or survival in rugged high-altitude mountain terrain.")  # 99 movies
    MUSIC = (121, "Music", "A movie focused on musicians, performances, concerts, or the music world, even if it is not a musical.")  # 3,708 movies
    MUSIC_DOCUMENTARY = (122, "Music Documentary", "A nonfiction film about musicians, music scenes, performances, genres, or music history.")  # 478 movies
    MUSICAL = (123, "Musical", "A movie where songs sung by the characters are woven into the story to express character or advance the plot.")  # 3,319 movies
    MYSTERY = (124, "Mystery", "A movie where solving a crime, disappearance, or hidden puzzle drives the story.")  # 10,050 movies
    NATURE_DOCUMENTARY = (125, "Nature Documentary", "A nonfiction film about wildlife, ecosystems, landscapes, or the natural world.")  # 100 movies
    NEWS = (126, "News", "A movie presented in a news or newsreel style that reports real current events or topical issues.")  # 172 movies
    NORWEGIAN = (127, "Norwegian", "A movie whose primary spoken language is Norwegian.")  # 251 movies
    ONE_PERSON_ARMY_ACTION = (128, "One-Person Army Action", "An action movie built around a lone hero taking on large numbers of enemies by themselves.")  # 533 movies
    PARANORMAL_REALITY_TV = (129, "Paranormal Reality TV", "A reality-style title where people supposedly investigate or experience ghosts, hauntings, or other paranormal phenomena.")  # 1 movie
    PARODY = (130, "Parody", "A comedy that imitates and exaggerates specific films, genres, or cultural targets to mock them humorously.")  # 1,293 movies
    PERIOD_DRAMA = (131, "Period Drama", "A drama set in a clearly defined past era where the historical setting shapes the story, manners, and look.")  # 2,126 movies
    PERSIAN = (132, "Persian", "A movie whose primary spoken language is Persian.")  # 219 movies
    POLICE_PROCEDURAL = (133, "Police Procedural", "A crime drama centered on police work, with emphasis on investigation, evidence, and official procedure.")  # 155 movies
    POLITICAL_DOCUMENTARY = (134, "Political Documentary", "A nonfiction film that examines, critiques, or advocates around political events, leaders, systems, or movements.")  # 84 movies
    POLITICAL_DRAMA = (135, "Political Drama", "A drama whose central conflict turns on politics, government, elections, or the exercise of public power.")  # 459 movies
    POLITICAL_THRILLER = (136, "Political Thriller", "A thriller driven by political power struggles, conspiracies, or state-level stakes.")  # 321 movies
    POP_MUSICAL = (137, "Pop Musical", "A musical whose songs and style are driven by pop music.")  # 129 movies
    PORTUGUESE = (138, "Portuguese", "A movie whose primary spoken language is Portuguese.")  # 569 movies
    PRISON_DRAMA = (139, "Prison Drama", "A drama set largely in prison and focused on incarceration, inmate life, and the conflicts it creates.")  # 180 movies
    PSYCHOLOGICAL_DRAMA = (140, "Psychological Drama", "A drama driven by characters' inner emotional and mental conflicts.")  # 3,028 movies
    PSYCHOLOGICAL_HORROR = (141, "Psychological Horror", "A horror movie that disturbs through paranoia, unstable minds, and emotional unease more than overt physical threats.")  # 763 movies
    PSYCHOLOGICAL_THRILLER = (142, "Psychological Thriller", "A thriller that creates suspense through obsession, unstable perceptions, or mental conflict more than physical danger.")  # 2,312 movies
    PUNJABI = (143, "Punjabi", "A movie whose primary spoken language is Punjabi.")  # 141 movies
    QUEST = (144, "Quest", "An adventure story built around a difficult journey toward a specific goal, object, or mission.")  # 424 movies
    QUIRKY_COMEDY = (145, "Quirky Comedy", "A comedy built around offbeat characters, eccentric behavior, and an oddball or whimsical tone.")  # 553 movies
    RAUNCHY_COMEDY = (146, "Raunchy Comedy", "A comedy that leans on explicit sexual humor, crude jokes, and deliberately risqué situations.")  # 477 movies
    REALITY_TV = (147, "Reality TV", "A title presented as unscripted or supposedly real-life situations featuring ordinary people rather than scripted actors.")  # 11 movies
    ROAD_TRIP = (148, "Road Trip", "A movie built around a long journey by road, where the trip itself drives the plot and changes the characters.")  # 269 movies
    ROCK_MUSICAL = (149, "Rock Musical", "A musical whose songs and overall sound are primarily rooted in rock music.")  # 49 movies
    ROMANCE = (150, "Romance", "A movie whose main plot centers on romantic love and the emotional relationship between the leads.")  # 20,161 movies
    ROMANTIC_COMEDY = (151, "Romantic Comedy", "A romance where the central love story is played for humor and emotional warmth, usually with a happy resolution.")  # 1,227 movies
    ROMANTIC_EPIC = (152, "Romantic Epic", "A sweeping love story set against major historical or social upheaval over an extended span of time.")  # 192 movies
    RUSSIAN = (153, "Russian", "A movie whose primary spoken language is Russian.")  # 901 movies
    SAMURAI = (154, "Samurai", "A movie centered on samurai warriors, sword fighting, and a historical Japanese setting.")  # 92 movies
    SATIRE = (155, "Satire", "A movie that uses humor, irony, exaggeration, or parody to mock and criticize society, institutions, or human folly.")  # 1,854 movies
    SCI_FI = (156, "Sci-Fi", "A movie where speculative science or technology such as aliens, robots, time travel, or space travel drives the story.")  # 7,671 movies
    SCI_FI_EPIC = (157, "Sci-Fi Epic", "A large-scale science-fiction story with expansive worlds, big ideas, and far-reaching stakes.")  # 219 movies
    SCIENCE_AND_TECHNOLOGY_DOCUMENTARY = (158, "Science & Technology Documentary", "A nonfiction film about scientific discoveries, technology, or their real-world effects.")  # 52 movies
    SCREWBALL_COMEDY = (159, "Screwball Comedy", "A comedy that uses fast dialogue, farcical misunderstandings, and chaotic romantic conflict.")  # 372 movies
    SEA_ADVENTURE = (160, "Sea Adventure", "An adventure whose journey, quest, or survival story unfolds primarily at sea or on other large bodies of water.")  # 202 movies
    SEINEN = (161, "Seinen", "An anime aimed at adult men and marked by more mature themes or complex storytelling.")  # 48 movies
    SERIAL_KILLER = (162, "Serial Killer", "A movie centered on a murderer who kills multiple victims in separate incidents over time, often following a pattern.")  # 281 movies
    SHORT = (163, "Short", "A short film, typically running 40 minutes or less including credits.")  # 5,646 movies
    SHOWBIZ_DRAMA = (164, "Showbiz Drama", "A drama centered on careers, rivalries, pressures, and personal struggles inside the entertainment industry.")  # 210 movies
    SHOJO = (165, "Shōjo", "An anime aimed at girls and young women, usually emphasizing emotions and relationships.")  # 19 movies
    SHONEN = (166, "Shōnen", "An anime aimed at adolescent boys, often emphasizing action, adventure, and coming-of-age struggle.")  # 62 movies
    SITCOM = (167, "Sitcom", "A comedy built around recurring characters in a shared setting, with humor from everyday situations and conflicts.")  # 11 movies
    SKETCH_COMEDY = (168, "Sketch Comedy", "A movie made up of short, self-contained comic scenes instead of one continuous story.")  # 77 movies
    SLAPSTICK = (169, "Slapstick", "A comedy that depends on exaggerated physical gags, pratfalls, and absurd action.")  # 1,820 movies
    SLASHER_HORROR = (170, "Slasher Horror", "A horror movie where a killer stalks and murders victims one by one, usually with sharp weapons and intense suspense.")  # 1,145 movies
    SLICE_OF_LIFE = (171, "Slice of Life", "A movie focused on ordinary daily experiences and relationships rather than high-stakes plot.")  # 26 movies
    SOAP_OPERA = (172, "Soap Opera", "A melodramatic story driven by ongoing personal conflict and interconnected relationship drama.")  # 1 movie
    SOCCER = (173, "Soccer", "A sports movie where soccer is central to the story and to the characters' struggles or goals.")  # 92 movies
    SPACE_SCI_FI = (174, "Space Sci-Fi", "A science-fiction movie set mainly in outer space or built around space travel, exploration, or interstellar conflict.")  # 552 movies
    SPAGHETTI_WESTERN = (175, "Spaghetti Western", "A Western made mainly by Italian filmmakers in Europe and marked by a grittier, more stylized tone.")  # 394 movies
    SPANISH = (176, "Spanish", "A movie whose primary spoken language is Spanish.")  # 2,874 movies
    SPLATTER_HORROR = (177, "Splatter Horror", "A horror movie defined by extreme, graphic gore, mutilation, and bloodshed shown on screen.")  # 393 movies
    SPORT = (178, "Sport", "A movie where a sport, athlete, team, or coach is central to the story.")  # 2,407 movies
    SPORTS_DOCUMENTARY = (179, "Sports Documentary", "A nonfiction film about athletes, teams, competitions, or the wider cultural impact of sports.")  # 182 movies
    SPY = (180, "Spy", "A movie centered on espionage, covert missions, intelligence work, and secret identities.")  # 366 movies
    STAND_UP = (181, "Stand-Up", "A movie built around stand-up comedy performances, with a comedian speaking directly to an audience.")  # 123 movies
    STEAMPUNK = (182, "Steampunk", "A movie set in a retro-futuristic world shaped by Victorian-style steam-era technology and alternate-history design.")  # 67 movies
    STEAMY_ROMANCE = (183, "Steamy Romance", "A romance strongly driven by sexual attraction and sensual encounters.")  # 240 movies
    STONER_COMEDY = (184, "Stoner Comedy", "A comedy built around marijuana use, stoner characters, and cannabis-fueled humor or misadventures.")  # 90 movies
    STOP_MOTION_ANIMATION = (185, "Stop Motion Animation", "An animated movie made by moving real objects or puppets in small increments and photographing them frame by frame.")  # 222 movies
    SUPERHERO = (186, "Superhero", "A movie about heroes with extraordinary powers, skills, or technology who fight villains and protect others.")  # 839 movies
    SUPERNATURAL_FANTASY = (187, "Supernatural Fantasy", "A fantasy movie where ghosts, spirits, monsters, or other supernatural forces are central to the story.")  # 492 movies
    SUPERNATURAL_HORROR = (188, "Supernatural Horror", "A horror movie where ghosts, demons, curses, possession, or other paranormal forces are the main source of fear.")  # 1,400 movies
    SURVIVAL = (189, "Survival", "A movie about characters struggling to stay alive in extreme, life-threatening conditions.")  # 319 movies
    SUSPENSE_MYSTERY = (190, "Suspense Mystery", "A mystery that emphasizes tension and danger as clues, secrets, and twists gradually reveal a central unknown.")  # 607 movies
    SWASHBUCKLER = (191, "Swashbuckler", "An adventure movie about a dashing sword-wielding hero, full of duels, daring exploits, and romance.")  # 227 movies
    SWEDISH = (192, "Swedish", "A movie whose primary spoken language is Swedish.")  # 521 movies
    SWORD_AND_SANDAL = (193, "Sword & Sandal", "A movie set in ancient or mythic antiquity, usually centered on warriors, rulers, gladiators, or legendary heroes.")  # 81 movies
    SWORD_AND_SORCERY = (194, "Sword & Sorcery", "A fantasy movie about sword-wielding heroes facing magic, monsters, and dangerous quests in a mythic world.")  # 162 movies
    TALK_SHOW = (195, "Talk Show", "A title built around a host interviewing or conversing with guests on entertainment, current events, or human-interest topics.")  # 3 movies
    TAMIL = (196, "Tamil", "A movie whose primary spoken language is Tamil.")  # 1,182 movies
    TEEN_ADVENTURE = (197, "Teen Adventure", "An adventure movie centered on teenage protagonists whose journeys or quests drive the story and their coming-of-age.")  # 161 movies
    TEEN_COMEDY = (198, "Teen Comedy", "A comedy focused on teenage characters and the humor of high school life, friendship, romance, and adolescence.")  # 431 movies
    TEEN_DRAMA = (199, "Teen Drama", "A drama focused on teenage characters and their relationships, identity, family conflicts, and coming-of-age struggles.")  # 462 movies
    TEEN_FANTASY = (200, "Teen Fantasy", "A fantasy movie about teenage characters whose coming-of-age is intertwined with magic, supernatural forces, or fantastical worlds.")  # 128 movies
    TEEN_HORROR = (201, "Teen Horror", "A horror story focused on teenagers confronting killers, monsters, or supernatural threats amid teen life.")  # 358 movies
    TEEN_ROMANCE = (202, "Teen Romance", "A romance focused on teenagers, usually about first love, dating, heartbreak, and the intensity of adolescence.")  # 245 movies
    TELUGU = (203, "Telugu", "A movie whose primary spoken language is Telugu.")  # 857 movies
    THAI = (204, "Thai", "A movie whose primary spoken language is Thai.")  # 243 movies
    THRILLER = (205, "Thriller", "A suspense-driven movie built around danger, tension, and uncertainty, designed to keep viewers on edge.")  # 23,577 movies
    TIME_TRAVEL = (206, "Time Travel", "A movie where characters travel to the past or future, often creating paradoxes, alternate timelines, or attempts to change history.")  # 224 movies
    TRAGEDY = (207, "Tragedy", "A serious story defined by suffering, downfall, or irreversible loss rather than triumph.")  # 1,094 movies
    TRAGIC_ROMANCE = (208, "Tragic Romance", "A romance where the central love story ends in heartbreak, separation, sacrifice, or death rather than a happy union.")  # 375 movies
    TRAVEL_DOCUMENTARY = (209, "Travel Documentary", "A nonfiction film structured around travel, using a journey to explore places, cultures, landscapes, and local experiences.")  # 66 movies
    TRUE_CRIME = (210, "True Crime", "A nonfiction crime story based on a real case, focusing on actual crimes, investigations, victims, perpetrators, and legal consequences.")  # 832 movies
    TURKISH = (211, "Turkish", "A movie whose primary spoken language is Turkish.")  # 518 movies
    URBAN_ADVENTURE = (212, "Urban Adventure", "An adventure story set chiefly in a city, where characters navigate urban spaces through danger, mystery, action, or discovery.")  # 252 movies
    URDU = (213, "Urdu", "A movie whose primary spoken language is Urdu.")  # 92 movies
    VAMPIRE_HORROR = (214, "Vampire Horror", "A horror story where vampires are the central source of fear.")  # 239 movies
    WAR = (215, "War", "A movie centered on warfare and its effects on soldiers, civilians, and the costs of armed conflict.")  # 4,171 movies
    WAR_EPIC = (216, "War Epic", "A large-scale war story with sweeping scope, historical weight, and the monumental impact of conflict.")  # 167 movies
    WATER_SPORT = (217, "Water Sport", "A sports story centered on activities in or on water, such as surfing, swimming, diving, rowing, or sailing.")  # 58 movies
    WEREWOLF_HORROR = (218, "Werewolf Horror", "A horror story where werewolves are the main source of terror, with transformations and lycanthropic attacks driving the fear.")  # 134 movies
    WESTERN = (219, "Western", "A story set in the 19th-century American West, usually involving frontier life, outlaws, lawmen, settlers, justice, and wilderness.")  # 3,697 movies
    WESTERN_EPIC = (220, "Western Epic", "A large-scale Western that combines frontier settings with sweeping scope, grand landscapes, and historically weighty storytelling.")  # 81 movies
    WHODUNNIT = (221, "Whodunnit", "A mystery structured around identifying the culprit, with clues inviting the audience to solve the crime before the reveal.")  # 693 movies
    WITCH_HORROR = (222, "Witch Horror", "A horror story where witches, witchcraft, or occult curses are the main source of fear and menace.")  # 67 movies
    WORKPLACE_DRAMA = (223, "Workplace Drama", "A drama centered on professional environments, where the main conflicts come from work relationships, ambition, power struggles, and job pressures.")  # 223 movies
    WUXIA = (224, "Wuxia", "A Chinese martial-arts adventure about chivalrous heroes, usually in historical settings with stylized combat and quests for justice.")  # 85 movies
    ZOMBIE_HORROR = (225, "Zombie Horror", "A horror story where zombies or zombie-like infected humans are the central threat, often driving survival or outbreak scenarios.")  # 333 movies


KEYWORD_BY_NORMALIZED_NAME: dict[str, OverallKeyword] = {
    normalize_string(keyword.value): keyword
    for keyword in OverallKeyword
}


def keyword_from_string(raw_keyword: str) -> OverallKeyword | None:
    """Resolve a raw keyword string to the canonical OverallKeyword enum."""
    normalized = normalize_string(raw_keyword)
    if not normalized:
        return None
    return KEYWORD_BY_NORMALIZED_NAME.get(normalized)
