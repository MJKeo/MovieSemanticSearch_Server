# Few-shot examples - Music / score acclaim

<example>
Input:
```xml
<retrieval_intent>Find films with iconic scores.</retrieval_intent>
<expressions><expression>iconic score</expression></expressions>
```
Expected: fire semantic reception; praised music/score qualities.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies with great soundtracks.</retrieval_intent>
<expressions><expression>great soundtrack</expression></expressions>
```
Expected: fire semantic reception; soundtrack praise.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films composed by John Williams.</retrieval_intent>
<expressions><expression>John Williams</expression></expressions>
```
Expected: no-fire; named composer credit belongs to person credit.
</example>

<example>
Input:
```xml
<retrieval_intent>Find musicals.</retrieval_intent>
<expressions><expression>musical</expression></expressions>
```
Expected: no-fire; genre/form, not score acclaim.
</example>

