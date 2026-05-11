# Few-shot examples

Each example shows a call that survived Step 2's compound-splitting and
landed at this category as a single atomic genre trait. The expected
outcome shows which endpoint commits and which abstains — never both.

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

<example>
Input:
```xml
<retrieval_intent>Identify films in the revenge story archetype.</retrieval_intent>
<expressions><expression>revenge stories</expression></expressions>
```
Expected: abstain BOTH endpoints. Revenge is a story shape, not a
genre identity; this call should have been routed to Story / thematic
archetype upstream. A misrouted call produces an honest double-abstain,
not a coerced commit on either side.
</example>
