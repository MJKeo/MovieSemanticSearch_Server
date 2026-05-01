# Few-shot examples

<example>
Input:
```xml
<retrieval_intent>Find films in a broad canonical genre.</retrieval_intent>
<expressions><expression>horror movies</expression></expressions>
```
Expected: preferred-only. Keyword directly covers the genre.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films in a specific named subgenre.</retrieval_intent>
<expressions><expression>body horror</expression></expressions>
```
Expected: preferred-only if the registry directly covers the subgenre;
otherwise fallback-only to Semantic genre_signatures. Do not substitute
plain horror when the named subgenre is uncovered.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with a qualified genre texture.</retrieval_intent>
<expressions><expression>quiet drama</expression></expressions>
```
Expected: split only if the base genre is covered and the qualifier
remains in this category call; otherwise fallback Semantic
genre_signatures for the qualified texture.
</example>

<example>
Input:
```xml
<retrieval_intent>Find revenge-shaped stories.</retrieval_intent>
<expressions><expression>revenge stories</expression></expressions>
```
Expected: no-fire. This is a story archetype, not genre identity.
</example>
