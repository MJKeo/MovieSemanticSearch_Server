# SourceMaterialType Enum — Finalized Definition

Derived by analyzing all 4,311 unique free-text `source_material` values from the
current `generated_metadata.source_of_inspiration` column in tracker.db. The top 100
values by movie count were clustered into distinct, non-overlapping groups that
represent meaningful user-facing search distinctions.

**Storage format:** `source_material_types: SourceMaterialType[]` — an array of enum
values per movie, because movies frequently have multiple applicable types (e.g.,
Schindler's List = `NOVEL_ADAPTATION` + `TRUE_STORY`, Bohemian Rhapsody = `BIOGRAPHY`
+ `TRUE_STORY`). The LLM should assign all that apply. An empty array means original
screenplay — the movie has no external source material.

**Original screenplay handling:** `ORIGINAL_SCREENPLAY` was removed from the enum
during implementation. Original screenplays are identified by an empty
`source_material_type_ids` array — this is a derivable signal, not something the LLM
needs to classify. Queries for "original screenplays" filter for movies where the
array is empty (`source_material_type_ids = '{}'`).

---

## Enum Values

### `NOVEL_ADAPTATION`

**Represents:** Adapted from a long-form prose work (fiction or non-fiction).

**Encompasses:** "based on a novel," "based on a book," "based on a novella," "based
on novels," "based on books," "based on a book series," "based on a series of novels,"
"based on a novel series," "based on a children's book," "based on a children's book
series," "based on a children's novel," "based on a young adult novel," "based on a
non-fiction book," "based on an autobiographical novel," "based on a light novel"

**Boundary note:** Light novels are included here (long-form prose), not under
`COMIC_ADAPTATION`, despite their association with manga/anime culture.

---

### `SHORT_STORY_ADAPTATION`

**Represents:** Adapted from a shorter written work — a piece that is not a full-length
book.

**Encompasses:** "based on a short story," "based on short stories," "based on a
story," "based on a magazine story," "based on a magazine article," "based on an
article," "based on a poem," "based on an epic poem"

**Boundary note:** Poems are included here rather than under `FOLKLORE_ADAPTATION`
because the distinguishing factor is the *literary format* — a poem by a known author
is a written work being adapted, whereas folklore is an oral/cultural tradition with
no single modern author. Epic poems that are primarily mythological (e.g., the Iliad)
will legitimately carry both this and `FOLKLORE_ADAPTATION`.

---

### `STAGE_ADAPTATION`

**Represents:** Adapted from a work originally performed on stage.

**Encompasses:** "based on a play," "based on a stage play," "based on a stage
musical," "based on a Broadway musical," "based on a musical," "based on an opera,"
"based on an operetta," "based on a ballet"

---

### `TRUE_STORY`

**Represents:** The film depicts real events. The focus is on *what happened*, not on
profiling a specific individual's life.

**Encompasses:** "based on true events," "based on a true story," "based on historical
events," "based on autobiographical material"

**Boundary with BIOGRAPHY:** A film can carry both. *Argo* is `TRUE_STORY` (the Iran
hostage rescue happened). *Bohemian Rhapsody* is `BIOGRAPHY` + `TRUE_STORY` (the
events depicted are real and it's about a specific person's life). The conceptual
split: `TRUE_STORY` = "this happened," `BIOGRAPHY` = "this is someone's life story."

---

### `BIOGRAPHY`

**Represents:** The film is primarily a portrait of a real person's life or a
significant portion of it.

**Encompasses:** "based on a real person," "based on real people," "based on a
biography," "based on an autobiography," "based on a memoir," "biopic," "biographical
film," "biographical documentary," "biography," "biopic (based on a real person),"
"documentary/biography"

**Boundary note:** "documentary" (157 count) appearing in the source_material field is
a format contamination issue — documentary is a film format, not a source type. During
re-generation, the LLM should not output a source material type for the mere fact that
a film is a documentary. Biographical documentaries about a real person still qualify
as `BIOGRAPHY`.

---

### `COMIC_ADAPTATION`

**Represents:** Adapted from a visual sequential-art medium — comics, graphic novels,
manga, or related formats.

**Encompasses:** "based on a comic book," "based on a comic strip," "based on a
comic," "based on a graphic novel," "based on a manga," "based on a manhwa," "based
on a comic strip character," "based on a cartoon character," "based on cartoon
characters"

**Boundary note:** "based on a cartoon character" (127) and "based on a literary
character" (83) are character-based entries where the source is really the originating
medium. During re-generation, the LLM should trace to the *originating medium* —
Sherlock Holmes -> `NOVEL_ADAPTATION`, Batman -> `COMIC_ADAPTATION`, Bugs Bunny ->
`TV_ADAPTATION`. The "character" phrasing should not appear in re-generated output.

---

### `FOLKLORE_ADAPTATION`

**Represents:** Adapted from oral tradition, mythology, religious scripture, or
culturally inherited stories with no single modern author.

**Encompasses:** "based on a fairy tale," "based on a folktale," "based on a folk
tale," "based on folklore," "based on a legend," "based on an urban legend," "based
on mythology," "based on a biblical story," "based on the Bible," "based on a
religious text"

**Boundary note:** Urban legends are included — they're modern oral tradition. A film
adapting the Iliad could carry both `SHORT_STORY_ADAPTATION` (epic poem as literary
form) and `FOLKLORE_ADAPTATION` (mythological origin). The LLM should apply both when
warranted.

---

### `VIDEO_GAME_ADAPTATION`

**Represents:** Adapted from a video game or video game franchise.

**Encompasses:** "based on a video game"

Small but culturally distinct and a common user search intent ("video game movies").

---

### `REMAKE`

**Represents:** A new version of a previously produced film or short film — same
essential story, re-executed.

**Encompasses:** "remake of a film," "remake," "reimagining of a film," "reimagining
of a novel," "adaptation of a film," "inspired by a film," "based on a film,"
"foreign-language adaptation," "live-action adaptation," "live-action adaptation of an
anime," "based on a short film," "remake of a French film," "remake of a TV series,"
"based on a film franchise"

**Boundary notes:**
- "based on a short film" (202) goes here — it's a previously produced film work being
  re-made/expanded into a feature.
- "remake of a TV series" (24) goes here, not `TV_ADAPTATION`, because the intent is
  "they remade X as a movie" — the remake relationship is the salient fact.
- "live-action adaptation of an anime" (31) goes here — the operative concept is
  remaking an existing screen work in a different format.
- A film can carry both `REMAKE` and another type. A live-action remake of a manga
  anime is `REMAKE` + `COMIC_ADAPTATION` (the original source is manga).

---

### `TV_ADAPTATION`

**Represents:** Adapted from serialized audio or visual media — television, web series,
radio, or animated series.

**Encompasses:** "based on a TV series," "based on a web series," "based on an anime,"
"based on a TV anime series," "based on a cartoon," "based on a cartoon series," "based
on a radio series," "based on a radio show," "based on a radio play"

**Boundary note:** Radio is included here rather than getting its own value. The shared
concept is "serialized broadcast/streaming media adapted to film." At 94 movies
combined, radio doesn't warrant a separate enum, and the user intent is similar —
"movies based on shows."

---

## Deliberate Exclusions

These categories were considered and intentionally excluded from the enum:

**Songs (68 movies):** Most "based on a song" movies are better described by their
other tags — a jukebox musical biopic is `BIOGRAPHY` + `STAGE_ADAPTATION`, a film
inspired by a single song has no applicable source material type (empty array). The
"song as source material" signal is too weak and ambiguous to be a useful filter.
During re-generation, these should be mapped to their more salient types.

**Toys (77 movies):** Toy-based films (Transformers, Barbie, G.I. Joe) are franchise IP
products — the franchise system (`movie_franchise_metadata`) captures this relationship
better than source material type. A toy line is not a narrative source the way a novel
or play is. During re-generation, these get an empty array (no source material type)
unless they also adapt a specific comic/TV series, in which case that type applies.

**Spinoff / Reboot (815 movies combined):** These are franchise_lineage concepts that
leaked into the free-text source_material field. Already handled by the `FranchiseRole`
enum. Should not appear in re-generated source_material output.

**Documentary (157 movies):** A film format, not a source type. Biographical
documentaries get `BIOGRAPHY`; event documentaries about real events get `TRUE_STORY`;
observational documentaries with no narrative source get an empty array.

**Parody (54 movies):** A film's *relationship* to its reference material, not a source
type. A parody of a film is still `REMAKE` in the source-material sense. The parody
nature is better captured elsewhere (genre, tone metadata).
