# Adaptation source flag — additional notes

This category is a **yes/no flag on the origin medium of the story** — novel, short story, stage, comic, folklore, video game, TV series, real events, one real person's life, or an earlier film. The emission is a single `UnifiedClassification` member drawn from the SourceMaterialType family (`NOVEL_ADAPTATION`, `SHORT_STORY_ADAPTATION`, `STAGE_ADAPTATION`, `TRUE_STORY`, `BIOGRAPHY`, `COMIC_ADAPTATION`, `FOLKLORE_ADAPTATION`, `VIDEO_GAME_ADAPTATION`, `TV_ADAPTATION`, `REMAKE`). Pick the member whose definition most directly names the origin medium the user has in mind.

## How to pick the right member

Match on the **medium of the source**, not on the subject matter depicted. "Based on a true story" names real events → `TRUE_STORY`. "About Lincoln's life" names one person's named life → `BIOGRAPHY`. "Based on a Stephen King novel" names a full-length prose source → `NOVEL_ADAPTATION`. "Mortal Kombat movie" names a video-game source → `VIDEO_GAME_ADAPTATION`. Stage adaptations, comic adaptations, and folklore adaptations follow the same rule — pick by the origin medium the query cites or clearly implies.

When the query names an author or franchise but not the medium, infer the medium the author is known for writing in: Stephen King, Jane Austen, Philip K. Dick → `NOVEL_ADAPTATION`. This handler only emits the medium flag; the author itself is a separate atom handled by Source-material author (Cat 30).

## REMAKE in this category

`REMAKE` here is an **origin-medium flag meaning "the film retells an earlier film"**, applied independently of any named franchise. Fire `REMAKE` when the query asks for retellings of earlier films in general ("1980s remakes", "Hollywood remakes of foreign films", "remakes of silent classics"). Do **not** fire when the remake phrasing is scoped to a named franchise ("the Scarface remake", "the Dune remake") — that is lineage positioning inside a franchise and belongs to Franchise / universe lineage (Cat 4). The discriminator is whether the query names a specific franchise that anchors the remake; if it does, Cat 4 owns the slice.

## Boundaries with nearby categories

- **Specific subject (Cat 6).** "Movies about JFK", "Titanic movie", "Vietnam War movies" describe a real subject IN the story, not the source medium. That is Cat 6, not here. "JFK biopic" composes Cat 5 (`BIOGRAPHY`) with Cat 6 (the subject) — emit your slice (`BIOGRAPHY`) and let the subject atom run its own handler.
- **Source-material author (Cat 30).** The author name itself ("Stephen King", "Neil Gaiman") is Cat 30's atom. When step 2 routes the adaptation-flag slice here, emit the medium (`NOVEL_ADAPTATION`) and stop. Do not try to encode the author.
- **Franchise / universe lineage (Cat 4).** Franchise membership alone ("Marvel movies", "DC films", "Harry Potter movies") is not an adaptation flag. No-fire here — Cat 4 owns it. The fact that the franchise happens to originate in comics or novels does not license firing `COMIC_ADAPTATION` / `NOVEL_ADAPTATION` on a bare franchise query.
- **Top-level genre (Cat 11).** "Biographical drama" composes a genre atom with a Cat 5 `BIOGRAPHY` atom. Emit your flag; let the genre atom run separately.

## When to no-fire

- **No medium signal in the atom.** The target requirement does not actually name or imply an origin medium. Upstream dispatch was wrong. Record the mismatch in `coverage_gaps` and return `should_run_endpoint: false`.
- **Franchise-scoped remake phrasing.** The query frames the remake inside a specific named franchise — Cat 4's lane. No-fire.
- **Subject-in-story framing without an adaptation cue.** "Movies about JFK" names the subject, not the source medium. No-fire unless the atom explicitly carries the biopic / true-story framing.
- **Self-contradictory modifier the flag cannot express.** Keep the parameters describing the target medium directly; the wrapper's `polarity` field handles negation. Never flip the classification to a different member to simulate "not an adaptation."

The tightest failure mode: firing a SourceMaterialType member when the atom is really about **what's in the story** rather than **where the story came from**. If you cannot point to an origin-medium cue in the atomic rewrite or parent fragment, no-fire.
