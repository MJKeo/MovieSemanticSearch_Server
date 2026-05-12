# Additional objective notes - Named source creator

Look up films credited to a named source author — the novelist,
short-story writer, comics creator, or playwright whose work the
film adapts. Stephen King, Tolkien, Shakespeare, Philip K. Dick,
Neil Gaiman, Jane Austen.

## What the model decides here

Only one thing: the `forms` list. Role, posting-table, and any
prominence concern are handled by code after parse — do not
deliberate about them. Skip prominence language; skip "predominant
role" reasoning. Your attention belongs entirely on enumerating
the credited name variants for the named author.

## Forms walk is the only thing that affects retrieval

Source authors are credited under variant strings across films —
middle initials present or absent, full given name vs. common
short form, occasional pseudonyms. A missed form silently drops
every film that uses it. Walk a handful of well-known adaptations
and enumerate the distinct credit strings.

Examples of variant pairs that belong as separate forms:
- "J.R.R. Tolkien" alongside "John Ronald Reuel Tolkien"
- "Philip K. Dick" alongside "Philip Kindred Dick"
- "Sir Arthur Conan Doyle" alongside "Arthur Conan Doyle"
- "Stephen King" alongside his Richard Bachman pseudonym when
  the source work was published under that name

Diacritic, casing, and punctuation variants do NOT belong —
ingest-side normalization collapses them.

## How retrieval works (for context only)

Forms match exactly against the writer posting table. A film is
either credited to one of the forms (score 1.0) or not (score
0.0). The writer table aggregates every IMDB writing credit —
screenplay, story, novel, characters, adaptation — into one
bucket, which is why a single forms walk catches authors of the
source work alongside any film whose screenplay they also wrote.

## Loose / "inspired by" adaptations

This endpoint cannot retrieve adaptations whose source author is
not on the IMDB writer credits. *The Lion King* (Hamlet-inspired),
*She's the Man* (Twelfth Night-inspired), and *Clueless* (Emma-
inspired) do not credit Shakespeare or Austen and will not match.
That is the right behavior — those films are not "Stephen King
movies" or "Shakespeare adaptations" in any retrievable sense.
