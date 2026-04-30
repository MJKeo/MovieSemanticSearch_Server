# Source-material author — additional notes

This category covers **lookups by the author of the source work a movie adapts** — the novelist, short-story writer, comic creator, or playwright whose original book / story / comic / play the film is based on. "Stephen King adaptations", "Jane Austen movies", "Tolkien films", "Philip K. Dick stories", "Neil Gaiman works". The ask must name a specific author whose source work has been adapted to film.

## Why Semantic is the only channel

Source-material authors are not film credits and are not indexed in any posting table — the indexed roles cover actor, director, writer (screenplay), producer, and composer of the film itself, not authors of the underlying work. The author's name does, however, surface routinely in the **plot_events synopsis prose** ("based on Stephen King's novel", "adapted from Jane Austen's 'Emma'"), and sometimes in `reception.reception_summary` (reviews cite the source author) or `plot_analysis.elevator_pitch` (identity capsules for adaptations). Semantic matching on those surfaces is the one honest retrieval path.

## How to populate the semantic payload

- `primary_vector: "plot_events"`. Synopses name the source author more consistently than any other ingest surface.
- `plot_events.plot_summary` — a short prose body that explicitly names the author and frames the film as an adaptation of their work ("a film adapted from Stephen King's novel, telling the story of..."). Name plus "adapted from" / "based on" is what makes the embedding discriminative; the name alone drifts into unrelated mentions.
- `reception` entry is worth adding when the author is a household name whose source work is routinely cited in reviews — populate `reception_summary` with a sentence naming the author and the adaptation ("a widely-reviewed adaptation of an author's novel, with critics comparing it to the source work"). Keep it a supporting entry.
- Leave `plot_analysis` out unless the author's name is so associated with a genre shape that the identity capsule would plausibly invoke them — this is rare; prefer no-fire over padding.

## Critical invariant — adaptation class is Cat 5's job, not yours

Step 2 co-emits a Cat 5 (Adaptation source) atom alongside this Cat 30 atom whenever the query says something like "Stephen King adaptations". Cat 5 will carry the NOVEL_ADAPTATION (or COMIC_ADAPTATION, TRUE_STORY, etc.) keyword filter. **Do not add a keyword filter, and do not try to encode the adaptation medium inside your semantic body.** Your slice is strictly "which movies' prose names this author?" — the medium filter comes in through the co-emitted Cat 5 atom at merge time. Encoding it here would double-count the same signal.

## Boundaries with nearby categories

- **Credit + title text (Cat 1).** If the named person is an indexed film credit — director, actor, screenwriter of the film, producer, composer — that is Cat 1's territory. The discriminator is whether the person wrote the source work or worked on the film itself. "Neil Gaiman wrote the screenplay for Coraline" → Gaiman as film writer → Cat 1. "Neil Gaiman adaptations" → the adapted author → here. Same name, different role.
- **Adaptation source flag (Cat 5).** Cat 5 carries the adaptation-medium flag ("novel adaptation", "comic book movie") via a keyword tag. Cat 30 carries the author's identity via semantic name-matching. They co-emit on queries like "Stephen King adaptations"; stay in your slice.
- **Specific subject (Cat 6).** Cat 6 is about a subject IN the story — a biopic whose subject IS the author would be Cat 6 (the film is about the author's life), not Cat 30 (a film adapting the author's work). The discriminator: is the film telling the author's story, or telling a story the author wrote?
- **Below-the-line creator (Cat 29).** Both Cat 29 and Cat 30 rely on semantic name-matching against ingest prose. Cat 29 is film-craft creators (cinematographer, editor, costume designer) and targets `reception`. Cat 30 is source-material authors and targets `plot_events`. The named person's role determines routing.

## When to no-fire

Return `should_run_endpoint: false` when:

- The named person is an indexed film credit (director, actor, screenwriter of the film, producer, composer) — dispatch was wrong; Cat 1 owns that.
- The query names an adaptation medium or category without naming any author ("novels adapted to film", "comic book movies", "video-game adaptations") — Cat 5 territory; no author name means no honest Semantic signal to embed here.
- The query names a person who is not a source-material author (a film-craft creator, a real-world subject the film is about, a critic) — `plot_events` prose would not carry that person as a source-work author.

No-fire with the coverage gap written plainly is always better than embedding a plot-summary body whose author attribution the user did not actually provide.
