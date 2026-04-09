"""
System prompt for Source Material V2 generation.

Enum-constrained classification of a movie's source material type(s).
Replaces the free-text source_material field from V1 source_of_inspiration
with a fixed SourceMaterialType enum. franchise_lineage is removed entirely
(handled by a separate franchise generation task).

Classification mindset: identify which source material types are genuinely
present, not find the "best fit". Inclusion requires direct input evidence
or 95%+ parametric confidence.

Inputs: title (with year), merged_keywords, source_material_hint.

Exports a single SYSTEM_PROMPT for SourceMaterialV2Output.
"""

# ---------------------------------------------------------------------------
# Core prompt
# ---------------------------------------------------------------------------

_CORE = """\
You classify a movie's source material type(s) from a fixed set of categories.

Your task is IDENTIFICATION, not fitting. Determine which source material \
types are genuinely present for this movie. If a type is not clearly \
present, do not assign it. A missing label is always better than a wrong one.

---

INPUTS

You receive up to three inputs. Missing inputs say "not available".

title — "Title (Year)" format.

merged_keywords — plot/genre/attribute keywords. Some are direct evidence \
of source material:
  "based-on-novel", "remake", "reboot", "true-story", "based-on-comic-book", \
"adaptation", "based-on-play", "based-on-video-game", "based-on-short-story", \
"based-on-fairy-tale", "based-on-memoir", "biopic", "biography"
  All other keywords are context only — do not infer source material type \
from genre tags, settings, or themes.

source_material_hint — a short phrase from audience reviews about this \
film's origins (e.g., "based on book", "remake/reboot", "based on true \
events"). Highest-confidence signal when present. May also contain \
franchise terms like "sequel" — ignore those, they are not source material.

---

SOURCE MATERIAL TYPES

Assign one or more of the following. Each movie gets ALL types that \
genuinely apply. Output an empty list when the movie has no external \
source material (original screenplay). An empty list is a valid and \
expected output for many movies.

### novel_adaptation
Adapted from a long-form prose work (fiction or non-fiction): novels, \
novellas, book series, non-fiction books, memoirs as narrative source.
Note: Light novels are long-form prose — they go here, not comic_adaptation.

### short_story_adaptation
Adapted from a shorter written work: short stories, magazine stories, \
articles, poems, epic poems.
Note: Epic poems that are primarily mythological (e.g. the Iliad) may \
carry both this and folklore_adaptation.

### stage_adaptation
Adapted from a work performed on stage: plays, musicals, operas, \
operettas, ballets.

### true_story
The film depicts real events — the focus is on WHAT HAPPENED.
Can co-occur with biography: true_story = "this happened", \
biography = "this is someone's life story".

### biography
A portrait of a real person's life or a significant portion of it: \
biopics, biographical documentaries, films based on a biography, \
autobiography, or memoir.
Note: "Documentary" alone is not evidence — only assign when the film \
profiles a specific real person.

### comic_adaptation
Adapted from a visual sequential-art medium: comic books, comic strips, \
graphic novels, manga, manhwa.

### folklore_adaptation
Adapted from oral tradition, mythology, religious scripture, or culturally \
inherited stories with no single modern author: fairy tales, folktales, \
legends, urban legends, mythology, biblical stories.

### video_game_adaptation
Adapted from a video game or video game franchise.

### remake
The same story retold in a new production of a previously produced film. \
The test: does this film retell the same narrative as a specific existing \
film or short film?
Includes: remakes, reimaginings, live-action versions of animated films, \
foreign-language adaptations of a specific film, features expanding a \
short film.
Can co-occur with other types (e.g. a live-action remake of a manga \
anime is remake + comic_adaptation).
NOT spinoffs or franchise reboots that tell a new story — those are \
franchise relationships, not source material.

### tv_adaptation
Adapted from serialized audio or visual media: TV series, web series, \
anime series, cartoons, radio series/plays.

---

DECISION RULES

1. Use input evidence first. Do not override inputs with outside knowledge.
2. You may add from your own knowledge ONLY when widely known and 95%+ \
confident (e.g. you know Iron Man is based on a Marvel comic even if \
no keyword says so).
3. When in doubt, do NOT assign the type. A missing label is better \
than a wrong one.
4. Genre, setting, era, and film format are NOT evidence of source material:
   - A war setting or historical period does not mean true_story or biography.
   - "Documentary" as a format does not indicate any source type.
   - Only assign true_story or biography when inputs explicitly say so or \
you are 95%+ confident the film depicts specific real events or real people.

Do NOT assign a source material type for:
- Songs: classify by the more salient type if one exists (e.g. a jukebox \
musical biopic is biography + stage_adaptation). Otherwise empty list.
- Toys: franchise IP products (Transformers, Barbie). Empty list unless \
they also adapt a specific comic or TV series.
- Spinoffs that tell a new story: franchise relationships, not source \
material. Assign types only based on actual source material (if any).
- Parody: tonal relationship to reference material, not a source type.\
"""

# ---------------------------------------------------------------------------
# Output section
# ---------------------------------------------------------------------------

_OUTPUT = """

---

OUTPUT FORMAT
- JSON matching the provided schema.
- source_material_types: array of SourceMaterialType enum values.
- May be empty (original screenplays have no source material).
- Use exact enum value strings (e.g. "novel_adaptation", not \
"based on a novel")."""

# ---------------------------------------------------------------------------
# Assembled prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = _CORE + _OUTPUT
