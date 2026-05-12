# Few-shot examples

Each example shows a call that survived Step 2's compound-splitting and
landed at this category as a single atomic trait. The expected outcome
shows which endpoint commits (or in Case B which spaces the semantic
body fires on) and which abstains.

## Case A — established genre identity

These calls name a registry-clean genre or a recognized sub-genre.
Exactly one endpoint commits; the other abstains.

<example>
Input:
```xml
<retrieval_intent>Identify films in the horror genre.</retrieval_intent>
<expressions><expression>scary movies</expression></expressions>
```
Expected: commit keyword with `[HORROR]`, scoring_method `ANY`. Abstain
semantic with `commitment-criteria-fail` — `HORROR` is definitionally
equivalent to "scary movies" (surface spelling differs, retrieval target
is identical), and semantic on the same slice would only inject scores
from genre-adjacent thrillers and dark dramas.
</example>

<example>
Input:
```xml
<retrieval_intent>Identify films in the spaghetti western sub-genre.</retrieval_intent>
<expressions><expression>spaghetti westerns</expression></expressions>
```
Expected: commit keyword with `[SPAGHETTI_WESTERN]`. Abstain semantic.
The sub-genre is its own canonical member; the parent `WESTERN` would
stretch over classical and contemporary westerns the user did not ask
for, and semantic adds no recall a clean tag hit can't already provide.
</example>

<example>
Input:
```xml
<retrieval_intent>Identify films in the dark comedy sub-genre.</retrieval_intent>
<expressions><expression>dark comedy</expression></expressions>
```
Expected: commit keyword with `[DARK_COMEDY]`. Abstain semantic. The
compound is a single canonical member — definitionally equivalent to
the user's phrase. Do NOT also commit `COMEDY` (would over-pull
mainstream comedy) or fire semantic on "dark" (the tonal modifier is
absorbed by the compound member; if the user wanted a distinct tonal
trait Step 2 would have split it out).
</example>

<example>
Input:
```xml
<retrieval_intent>Identify films in the neo-noir sub-genre.</retrieval_intent>
<expressions><expression>neo-noir</expression></expressions>
```
Expected: commit semantic with `plot_analysis.genre_signatures =
["neo-noir"]` and no other sub-fields populated. Abstain keyword with
`commitment-criteria-fail` — the registry carries `FILM_NOIR` but not
neo-noir, and they are DIFFERENT movements (classical 40s/50s vs the
70s-onward revival; different conventions, different films).
Committing `FILM_NOIR` would tag-match the wrong films at 1.0 while
true neo-noirs (Drive, LA Confidential, Blade Runner) score 0 on the
keyword channel. The user's phrase appears verbatim in the
`genre_signatures` ingest text for true neo-noirs and lands cleanly
there.
</example>

<example>
Input:
```xml
<retrieval_intent>Identify films in the elevated horror sub-genre.</retrieval_intent>
<expressions><expression>elevated horror</expression></expressions>
```
Expected: commit semantic with `plot_analysis.genre_signatures =
["elevated horror"]`. Abstain keyword with `commitment-criteria-fail`.
No canonical member definitionally covers "elevated horror" —
committing `HORROR` would stretch across the genre's full breadth
(slashers, creature features, schlock) at 1.0, which is exactly the
films the modifier "elevated" excludes. The user's verbatim phrase
embeds against the genre_signatures field, which uses critic
vocabulary that includes this term for the right films.
</example>

<example>
Input:
```xml
<retrieval_intent>Identify films in the cosmic horror sub-genre.</retrieval_intent>
<expressions><expression>cosmic horror</expression></expressions>
```
Expected: commit semantic with `plot_analysis.genre_signatures =
["cosmic horror"]`. Abstain keyword — `SUPERNATURAL_HORROR` and
`MONSTER_HORROR` both overlap cosmic horror partially but neither
definitionally covers it; firing either would over-pull supernatural
or creature features that are not cosmic horror.
</example>

## Case B — pseudo-genre description

These calls describe a film cluster that isn't a registry or canonical
sub-genre name. Keyword always abstains; semantic always commits and
authors across the eligible spaces. `plot_analysis` stays empty in every
Case B commit — that's the load-bearing signal of the Case B branch.

<example>
Input:
```xml
<retrieval_intent>Identify films whose look matches the comic book visual style.</retrieval_intent>
<expressions><expression>comic book visual style</expression></expressions>
```
Expected: commit semantic with `production.production_techniques`
naming the concrete craft choices the cluster is defined by
("comic-panel framing", "stylized color grading"), plus
`viewer_experience.sensory_load` for the perceptual texture (`terms`
like "vibrant", "stylized", "high-contrast"; `negations` like "not
muted", "not naturalistic"). `plot_analysis.genre_signatures` stays
empty — the phrase names a visual axis, not a genre identity. Abstain
keyword with `commitment-criteria-fail` — `COMIC_BOOK_ADAPTATION`
would tag-match every comic-book adaptation at 1.0 regardless of
visual style, which is exactly the breadth the user is trying to
narrow past.
</example>

<example>
Input:
```xml
<retrieval_intent>Identify films that scratch the slasher itch for viewers seeking that experience.</retrieval_intent>
<expressions><expression>scratches the slasher itch</expression></expressions>
```
Expected: commit semantic with `watch_context.self_experience_motivations`
naming the viewer appetite ("scratch the slasher itch",
"stalker-and-victim thrill"), `watch_context.key_movie_feature_draws`
for the cluster's signature draws ("kill set-pieces", "final-girl
tension"), and `viewer_experience.tension_adrenaline` +
`emotional_palette` for the dread-and-thrill texture
(`tension_adrenaline.terms` like "white-knuckle stalking", "kill-by-kill
escalation"). `plot_analysis.genre_signatures` stays empty. Abstain
keyword — firing `SLASHER_HORROR` would commit the canonical sub-genre
identity, but the call is shaped around the viewer's appetite for that
experience, not the genre identity itself; tag-matching every slasher
at 1.0 over-pulls toward the breadth of the sub-genre rather than the
appetite shape the user described.
</example>

<example>
Input:
```xml
<retrieval_intent>Identify films known for their noir-style voiceover storytelling.</retrieval_intent>
<expressions><expression>celebrated noir voiceover storytelling</expression></expressions>
```
Expected: commit semantic with `reception.praised_qualities` for the
acclaim aspect ("voiceover narration", "hardboiled voiceover",
"noir-style narration") and `narrative_techniques` with the canonical
craft label verbatim ("voiceover narrator"). `plot_analysis.genre_signatures`
stays empty. Abstain keyword — `FILM_NOIR` would tag-match the genre
identity, but the call is pointing at a craft-and-acclaim aspect of
noir-style films, not noir identity itself; tag-matching every film
noir at 1.0 would pull films without notable voiceover craft alongside
the ones the user wants.
</example>
