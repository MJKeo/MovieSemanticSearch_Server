# Top-level genre — additional notes

This category owns the broad-bucket genre question — horror, action, comedy, sci-fi, drama, romance, animation, thriller, musical, western, fantasy. The coarse top-level label, not a specific sub-genre. Exactly one of Keyword or Semantic fires; they answer different versions of the question, so stacking them would mix answers.

## What each endpoint answers here

- **Keyword** answers "is this one of the canonical top-level genres the registry already names as a member?" The registry holds a dedicated entry for every bare top-level genre (HORROR, ACTION, COMEDY, DRAMA, ROMANCE, ANIMATION, SCI_FI, MUSICAL, FANTASY, ROMANTIC_COMEDY, and so on). When the user names a bare canonical genre, Keyword resolves onto that member as a closed-enum GIN lookup — the most precise and recall-complete signal available.
- **Semantic plot_analysis / genre_signatures** answers "what genre-shaped *feel* is the user after when the qualifier on the genre can't be expressed as a tag?" The `genre_signatures` sub-field inside plot_analysis carries short genre-descriptor phrases on the ingest side ("dark action", "quiet drama", "heartwarming sci-fi", "slow-burn thriller"). A qualifier-laden compound lands there in the space's native vocabulary — as a short phrase or handful of phrases that read like the ingest-side signature of a matching movie.

## The discriminator is the qualifier

- **Bare canonical genre → Keyword.** "Horror movies", "romcoms", "westerns", "fantasy films". The label *is* the requirement; the registry already covers it.
- **Qualifier-laden compound that the tag vocabulary cannot express → Semantic.** "Dark action", "quiet drama", "heartwarming sci-fi", "slow-burn thriller". The tag for the bare genre would overshoot in one direction (return every horror regardless of whether it feels dark) and the registry has no tag for the qualified texture. Semantic plot_analysis captures the texture in the space's native vocabulary.
- **Named sub-genre → no-fire here, belongs to Cat 15.** "Body horror", "cozy mystery", "neo-noir", "space opera", "slasher", "giallo". These are identifiable sub-genre labels, not top-level genres with qualifiers. Route them to Cat 15 by emitting `endpoint_to_run: "None"` and naming the mismatch.

## Semantic routing when Semantic fires

When Semantic is the pick, **plot_analysis is the primary target** — specifically the `genre_signatures` sub-field, which is exactly where qualified genre textures live on the ingest side. Keep the body tight: a short phrase or two directly restating the qualified-genre compound in the space's native register. Other spaces (viewer_experience for felt tone, narrative_techniques for pacing) are populated only when the qualifier genuinely spans those dimensions — do not pad spaces to look thorough.

## Boundaries with nearby categories

- **Sub-genre + story archetype (Cat 15).** A named sub-genre ("body horror", "cozy mystery", "slow-burn thriller" as a recognized sub-genre label, "slasher") is Cat 15. A top-level genre with an open-ended qualifier ("dark action", "quiet drama") is Cat 11. The discriminator: is the label a recognized sub-genre name, or a bare top-level genre with a descriptive qualifier? Cat 11 handles the latter; route the former away.
- **Structured metadata (Cat 10).** Cat 10 handles numeric / closed-schema attribute predicates (release date, runtime, rating, reception score, streaming). Genre is never metadata here — do NOT emit METADATA. The Keyword genre vocabulary already covers what META.genre_ids would; adding metadata would be redundant at best and misroute at worst.
- **Viewer experience / feel (Cat 22).** Pure tone without a genre label ("dark", "moody", "cozy" on their own) is Cat 22 — Semantic viewer_experience. Cat 11 requires the genre label to be present; the qualifier shapes *which* genre texture, it is not the requirement on its own.

## When to no-fire

Return `endpoint_to_run: "None"` when:

- The requirement names a sub-genre rather than a top-level genre — Cat 15 owns that.
- The requirement is a bare qualifier with no genre label attached — Cat 22 or Cat 27 owns that.
- The phrasing is too vague to resolve onto a canonical member or author a genre_signatures body around ("movies with genre", "good genre films").
- The modifiers flip the requirement against itself in a way neither channel can express.

No-fire is always better than forcing a Keyword pick on a genre the user did not actually name, or authoring a Semantic genre_signatures body from an invented qualifier.
