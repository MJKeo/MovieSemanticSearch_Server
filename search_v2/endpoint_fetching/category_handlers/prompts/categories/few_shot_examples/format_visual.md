# Few-shot examples

<example>
Input:
```xml
<retrieval_intent>Find films in a canonical nonfiction format.</retrieval_intent>
<expressions><expression>documentary</expression></expressions>
```
Expected: preferred-only when the registry directly covers the format.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with a specific visual presentation feature.</retrieval_intent>
<expressions><expression>shot in black and white</expression></expressions>
```
Expected: preferred-only if a direct visual-format tag exists; otherwise
fallback-only to Semantic production. Do not treat as visual praise.
</example>

<example>
Input:
```xml
<retrieval_intent>Find documentary-form films with a specific shooting technique.</retrieval_intent>
<expressions><expression>documentary shot on 16mm</expression></expressions>
```
Expected: split if documentary is tagged and 16mm is not; Keyword covers
format, Semantic production covers technique.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films praised for visual beauty.</retrieval_intent>
<expressions><expression>visually stunning</expression></expressions>
```
Expected: no-fire. Acclaim belongs to Visual craft acclaim.
</example>
