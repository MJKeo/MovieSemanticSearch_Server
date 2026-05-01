# Few-shot examples - Title text lookup

<example>
Input:
```xml
<retrieval_intent>Find movies with the word Star in the title.</retrieval_intent>
<expressions><expression>Star</expression></expressions>
```
Expected: fire title-pattern lookup; contains match for "Star".
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies whose title starts with The Last.</retrieval_intent>
<expressions><expression>The Last</expression></expressions>
```
Expected: fire title-pattern lookup; starts-with match for "The Last".
</example>

<example>
Input:
```xml
<retrieval_intent>Find the exact movie title Inception.</retrieval_intent>
<expressions><expression>Inception</expression></expressions>
```
Expected: fire title-pattern lookup only if this exact-title request reaches
the handler; exact-match.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Star Wars movies.</retrieval_intent>
<expressions><expression>Star Wars</expression></expressions>
```
Expected: no-fire; this is franchise identity, not title text lookup.
</example>

