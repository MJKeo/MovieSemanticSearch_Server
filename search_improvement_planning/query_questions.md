# Query Questions

A composition-level breakdown of the kinds of questions a user's
natural-language query can be asking. Organized by how objectively
the user expects each question to be answered:

- **Objective** — one clear correct answer, no interpretation
- **Murky** — factual at the core but fuzzy at the edges
- **Subjective** — interpretation or consensus required

Two classes of query are **not** in this list because they are
flow-selection problems handled separately from query composition:

- Exact-title lookup (e.g. "Shawshank redemption")
- Holistic reference-anchored similarity (e.g. "movies like Inception")

**Principle — never guess a title from attributes.** Even when a
query strongly implies a specific movie via plot detail ("Indiana
Jones movie where he runs from the boulder"), the system always
searches by attributes rather than guessing a canonical title.
Well-scored attribute retrieval surfaces the intended movie without
risking a confident wrong guess.

---

## Objective — one clear correct answer

### 1. Who was involved in making it?
Presence/absence of a specific real person in a production credit
(actor, director, writer, producer, etc.).
*E.g.* "Steven Spielberg films", "starring Tom Hanks", "directed by
Kubrick", "not Adam Sandler", "written by Aaron Sorkin"

### 2. Who filled specific below-the-line creative credits?
Finer-grained non-actor/director production credits — composer,
cinematographer, editor, production designer. A sub-case of "who
was involved," but users often ask at this specific grain.
*E.g.* "Roger Deakins movies", "Hans Zimmer scores", "Thelma
Schoonmaker-edited", "Jonny Greenwood soundtracks"

### 3. What studio or brand produced it?
Movies from a specific production entity.
*E.g.* "Disney classics", "A24 films", "Pixar", "Studio Ghibli",
"Blumhouse horror"

### 4. What named character appears?
Fictional character presence as a fact (centrality handled separately
in murky).
*E.g.* "Batman movies", "any Wolverine appearance", "films with
Hermione"

### 5. What top-level genre is it?
Broad genre bucket at the coarse level (sub-genre handled separately
in murky).
*E.g.* "horror", "action films", "sci-fi", "romance", "animation"

### 6. When was it released?
Calendar date or era of release to audiences.
*E.g.* "80s movies", "released in 2019", "pre-1960", "last 5 years"

### 7. How long is it?
Runtime in minutes.
*E.g.* "under 90 minutes", "short movies", "three-hour epic"

### 8. What's the maturity rating?
Formal content rating (MPA / equivalent).
*E.g.* "PG-13", "R-rated", "G-rated", "kid-safe rating"

### 9. What audio language(s) is it in?
Spoken language(s) of the dialogue.
*E.g.* "Spanish-language", "in English", "dubbed", "foreign-language"

### 10. Where can I watch it?
Streaming platform availability.
*E.g.* "on Netflix", "on Hulu", "Prime", "available to stream"

### 11. What budget scale was it made at?
Numeric production budget tier (user's qualitative phrasing softened
by code).
*E.g.* "indie", "big-budget", "low-budget", "blockbuster"

### 12. How did it do at the box office?
Commercial performance.
*E.g.* "box office hits", "bombs", "highest-grossing"

### 13. Is it trending right now?
Current popularity snapshot.
*E.g.* "trending", "popular right now", "what everyone's watching"

### 14. What formal awards did it win or get nominated for?
Formal award status across real-world ceremonies and categories.
*E.g.* "Oscar winners", "Best Picture nominees", "Palme d'Or",
"Golden Globe-nominated"

### 15. Is it in a curated canon or critics' list?
Membership on a curator-selected list. Mechanism distinct from
formal awards (which are voted or juried).
*E.g.* "Criterion Collection", "AFI Top 100", "Sight & Sound
greatest", "IMDb Top 250", "BFI list"

### 16. What's its numeric reception score?
IMDB rating / Rotten Tomatoes percentage / equivalent as a number.
*E.g.* "rated above 8", "90%+ on Rotten Tomatoes"

### 17. What format is it?
Form factor of the work itself.
*E.g.* "short films", "documentaries", "mockumentary", "anime",
"silent films"

### 18. What visual-format specifics does it have?
Presentation-level visual format: color mode, capture medium, aspect
ratio, specific presentation style.
*E.g.* "black and white", "shot on film", "found-footage style",
"70mm", "handheld camerawork", "widescreen"

### 19. What's in the title text?
Lexical pattern in the title string.
*E.g.* "movies with 'love' in the title", "titles starting with 'The'"

### 20. What country was it produced in?
Production country as a legal/financial fact (cultural tradition
handled separately in murky).
*E.g.* "made in Japan", "French productions"

### 21. Where was the film physically shot?
Actual filming location(s) — distinct from production country (legal
origin) and narrative setting (in-story location).
*E.g.* "filmed in New Zealand", "shot on location in Iceland",
"in-studio vs on location", "shot in Morocco"

---

## Murky — factual at the core, fuzzy at the edges

### 1. Does it belong to a franchise or universe?
Membership in a named series or shared continuity; fuzzy at
crossover/spinoff edges.
*E.g.* "MCU", "Star Wars", "Pokemon movies", "Middle-earth"

### 2. Where does it sit in the franchise lineage?
Sequel / prequel / spinoff / remake / reboot / original positioning.
*E.g.* "not sequels", "spinoffs only", "the original not the remake",
"mainline Harry Potter", "first in a franchise"

### 3. What cultural tradition does it belong to?
National cinema or cultural tradition — often diverges from literal
production country.
*E.g.* "Korean cinema", "French films", "Bollywood", "Hong Kong
action", "Italian neorealism"

### 4. What specific sub-genre is it?
Fine-grained genre positioning — individual labels are sharp,
boundaries between labels blur.
*E.g.* "psychological thriller", "cozy mystery", "body horror",
"screwball comedy", "space opera", "spaghetti western", "neo-noir"

### 5. When is the story set (narrative time)?
The in-universe time period the story takes place in — distinct from
release date. A period drama is made now but set then.
*E.g.* "period drama set in the 1940s", "medieval setting", "Cold War
era", "wartime", "Victorian", "set during the Civil War"

### 6. Where is the story set (narrative location)?
The in-universe location the story takes place in — distinct from
production country.
*E.g.* "set in space", "takes place in Tokyo", "New York story", "set
in the American South", "high-school setting", "workplace", "small
town", "in the desert"

### 7. What specific things happen in it?
Concrete plot events — individually factual but often vaguely worded,
and sometimes used by the user as an implicit title-identifier.
*E.g.* "movies with a heist", "character fakes their death", "hotel
shootout", "plane crash survivors", "he runs from a boulder"

### 8. What's it adapted from or based on?
Source material flag — whether the movie is derived from some prior
work. Solid at the core, fuzzy at "based on true events" / "loosely
inspired by" / "remake vs reimagining."
*E.g.* "book adaptations", "based on a true story", "comic book
movies", "biographies", "remakes", "video-game adaptation"

### 9. What specific real-world people or events does it depict?
The subject the movie is *about* — distinct from "based on a true
story" (an adaptation flag) and from person-credit (who *made* it).
*E.g.* "about JFK", "about the Vietnam War", "about the Titanic",
"about Princess Diana", "Watergate movie"

### 10. What narrative devices does it use?
Structural/storytelling mechanics as discrete presence. Usually binary
but people disagree on what counts as each device.
*E.g.* "with a plot twist", "nonlinear timeline", "unreliable
narrator", "open ending", "fourth-wall break", "single-location"

### 11. What story archetype does it follow?
Recognizable story-engine pattern.
*E.g.* "revenge movies", "underdog stories", "con-artist",
"post-apocalyptic", "ensemble casts", "female-lead", "anti-hero"

### 12. What's the cast composition / ensemble shape?
The *shape* of the cast as a group, rather than who specifically is
in it.
*E.g.* "ensemble cast", "all-female cast", "two-hander", "unknown
actors", "stacked A-list cast"

### 13. How prominent is a given character?
Presence is objective; centrality is judgment (already softened in
code). Usually implicit in how the user phrases a character request.
*E.g.* implicit in "Batman movies" (Batman-centric, not cameos),
"Wolverine-heavy", "lone female protagonist"

### 14. How "old" or "new" does it feel?
Era as a qualitative band with fuzzy edges.
*E.g.* "classics", "old", "modern", "new releases", "recent"

### 15. What potentially sensitive content does it contain?
Content presence for warnings. Core presence is mostly binary;
intensity is murkier.
*E.g.* "no gore", "with nudity", "animal death", "violent", "not too
bloody"

### 16. Is it a holiday / seasonal movie?
Seasonal anchoring.
*E.g.* "Christmas movies", "Halloween", "Thanksgiving", "summer
blockbusters"

### 17. Who is it pitched to (audience / life-stage)?
Target demographic / age-band.
*E.g.* "family movies", "teen movies", "coming-of-age", "for adults"

---

## Subjective — interpretation or consensus required

### 1. What kind of story is it?
What it's fundamentally *about* — themes, character arcs, conflict
type, moral pitch.
*E.g.* "movies about grief", "tales of ambition", "stories about
forgiveness", "redemption arcs", "coming-of-age about self-acceptance"

### 2. How thematically weighty is it — does it have "something to say"?
Whether the movie has substantive thematic content vs. being pure
entertainment. Independent of whether themes exist — this is about
their *weight*.
*E.g.* "something to say", "has a message", "meaningful",
"thought-provoking", "makes you think", "substantive"

### 3. Is it character-focused or plot-focused?
Structural axis of storytelling — does the film foreground interiority
and relationships, or events and stakes?
*E.g.* "character study", "character-driven", "plot-heavy",
"event-driven", "intimate focus on one person"

### 4. How is the story told?
Craft and storytelling style beyond discrete devices.
*E.g.* "slow burn", "meditative pacing", "tightly plotted",
"frenetic", "dreamlike", "anthology-style", "slow-moving"

### 5. What's the realism / stylization mode?
Grounded-vs-heightened axis, orthogonal to genre and tone — how the
movie represents its world.
*E.g.* "grounded", "hyper-realistic", "naturalistic", "stylized",
"heightened reality", "over-the-top"

### 6. What's the scale and scope?
Magnitude and ambition of the storytelling — separate from runtime
or budget.
*E.g.* "intimate", "epic", "sprawling", "contained", "grand in
scope", "small and personal"

### 7. What does it feel like to watch?
Emotional / sensory / cognitive viewing experience.
*E.g.* "edge of your seat", "cozy", "gut-punch", "emotionally
devastating", "feel-good", "cathartic", "cry in a happy way",
"hits you in the gut"

### 8. How much does it demand from the viewer?
Cognitive-demand / depth axis — how much work it asks of the
audience.
*E.g.* "mindless", "popcorn", "easy watching", "undemanding" vs.
"cerebral", "heavy", "intellectually demanding", "dense"

### 9. How much does it linger afterward?
Post-viewing resonance — how durable the impact is after the credits
roll.
*E.g.* "stays with you", "lingers", "can't stop thinking about it",
"haunting", "forgettable"

### 10. How much rewatch value does it have?
Whether repeat viewings reward you. Distinct from "still holds up
today" (era aging) and "lingers afterward" (first-viewing aftermath).
*E.g.* "rewatchable", "holds up on rewatch", "one-and-done", "gets
better every time"

### 11. When / why would I watch this?
Viewing occasion / use-case.
*E.g.* "date night", "after a bad day", "background while cooking",
"something to watch high", "Sunday afternoon", "movie night with the
kids"

### 12. Is it a gateway / entry-level work for its genre?
Scaffolding — where the movie sits on the genre-appreciation curve.
*E.g.* "good first anime", "entry-level horror", "easy intro to slow
cinema", "accessible arthouse"

### 13. Is it a comfort-watch archetype?
Recommendation-archetype label combining rewatchability, low-stakes,
and familiar-feel — a specific viewing mode.
*E.g.* "comfort watch", "go-to movie", "background comfort",
"feel-better movie", "cozy favorite"

### 14. How is it crafted visually / technically?
Subjective sense of visual production craft and achievement.
*E.g.* "visually stunning", "beautifully shot", "unique aesthetic",
"killer cinematography", "technical marvel"

### 15. How is the music / score?
Musical craft as its own axis, independent of visual craft.
*E.g.* "iconic score", "great soundtrack", "killer needle drops",
"uses classical music", "memorable theme"

### 16. What's the dialogue quality / style?
Writing at the line level — distinct from overall story craft, tonal
aesthetic, or how-told.
*E.g.* "great dialogue", "quotable", "naturalistic dialogue",
"Sorkin-style", "quippy", "terse"

### 17. How did people respond to it?
Qualitative reception beyond formal awards or numeric scores.
*E.g.* "critically acclaimed", "cult classic", "underrated",
"universally loved", "divisive", "overhyped"

### 18. What's its cultural influence or historical importance?
The movie's *place* in cinematic history — distinct from reception
(how people responded) and awards (formal recognition).
*E.g.* "era-defining", "invented the genre", "most influential",
"changed cinema", "started a trend"

### 19. Does it still hold up today?
Temporal relevance — judgment of an older work through a contemporary
lens. Distinct from historical reception.
*E.g.* "still holds up", "worth watching today", "timeless", "hasn't
aged well", "dated"

### 20. How good is it on some axis?
Qualitative superlative ranking.
*E.g.* "best", "scariest", "funniest", "most thought-provoking",
"greatest of all time"

### 21. What tonal aesthetic does it have?
Tone / mood fingerprint — what the movie *is* rather than what
watching it *does*. Often genre-independent.
*E.g.* "dark and brooding", "neon noir", "whimsical", "bleak",
"gritty", "lighthearted", "wholesome", "moody", "soulful"
