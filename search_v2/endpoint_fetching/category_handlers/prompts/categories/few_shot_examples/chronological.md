# Few-shot examples - Chronological ordinal

<example>
Input:
```xml
<retrieval_intent>Prefer the newest film in the scoped set.</retrieval_intent>
<expressions><expression>newest</expression></expressions>
```
Expected:
- `direction`: `newest_first`

"Newest" → user wants the NEWEST movie on top → `newest_first`.
Newest movie scores 1.0; oldest scores 0.0; intermediate movies fall
on a continuous percentile curve.
</example>

<example>
Input:
```xml
<retrieval_intent>Prefer the oldest Kubrick films.</retrieval_intent>
<expressions><expression>oldest</expression></expressions>
```
Expected:
- `direction`: `oldest_first`

"Oldest" → user wants the OLDEST movie on top → `oldest_first`.
Sibling PERSON_CREDIT defines the pool — this endpoint does not
re-emit the director slice.
</example>

<example>
Input:
```xml
<retrieval_intent>Prefer the most recent Marvel movie.</retrieval_intent>
<expressions><expression>most recent</expression></expressions>
```
Expected:
- `direction`: `newest_first`

"Most recent" → user wants the NEWEST movie on top → `newest_first`.
Franchise slice is sibling, not here.
</example>

<example>
Input:
```xml
<retrieval_intent>Sort movies based on how recent they were produced.</retrieval_intent>
<expressions><expression>first</expression></expressions>
```
Expected:
- `direction`: `newest_first`

Conflict resolution: the expression "first" alone could read as
earliest, but the intent ("how recent") is a much more specific
date-direction signal — recency dominates. "First" here is
shorthand for "the topmost result of the recency sort," not a
preference for the earliest movie. Fire `newest_first` rather than
abstaining; the curve degrades gracefully even when inputs are
imperfectly aligned.
</example>

<example>
Input:
```xml
<retrieval_intent>Prefer the earliest Star Wars film.</retrieval_intent>
<expressions><expression>earliest</expression></expressions>
```
Expected:
- `direction`: `oldest_first`

"Earliest" → user wants the EARLIEST (= oldest) movie on top →
`oldest_first`. "First" / "earliest" / "original" all map here when
used as a date-position cue.
</example>

<example>
Input:
```xml
<retrieval_intent>Prefer recent movies.</retrieval_intent>
<expressions><expression>recent</expression></expressions>
```
Expected:
- `direction`: `newest_first`

"Recent" is a date-direction preference — newer movies should win.
Even though Cat 13 (RELEASE_DATE) may also fire for the same word
with a window framing, this endpoint contributes a continuous
direction signal and should fire alongside it. Don't abstain just
because the wording overlaps with another category.
</example>
