# Few-shot examples - Person credit

Use these for shape calibration, not as a closed list.

<example>
Input:
```xml
<retrieval_intent>Find films starring Tom Hanks.</retrieval_intent>
<expressions><expression>Tom Hanks</expression></expressions>
```
Expected: fire entity as person; role actor; prominence lead/starring if
available; no title or character interpretation.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films directed by Kathryn Bigelow.</retrieval_intent>
<expressions><expression>Kathryn Bigelow</expression></expressions>
```
Expected: fire entity as person; role director; one target for Kathryn
Bigelow.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with scores composed by John Williams.</retrieval_intent>
<expressions><expression>John Williams</expression></expressions>
```
Expected: fire entity as person; role composer.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films shot by Roger Deakins.</retrieval_intent>
<expressions><expression>Roger Deakins</expression></expressions>
```
Expected: no-fire; cinematographer is not an indexed role for this endpoint.
</example>

