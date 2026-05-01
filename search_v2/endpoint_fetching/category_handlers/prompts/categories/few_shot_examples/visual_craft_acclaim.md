# Few-shot examples - Visual craft acclaim

<example>
Input:
```xml
<retrieval_intent>Find films praised for cinematography.</retrieval_intent>
<expressions><expression>praised cinematography</expression></expressions>
```
Expected: fire semantic reception; praised_qualities includes
cinematography / visual composition.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies known for practical creature effects.</retrieval_intent>
<expressions><expression>practical creature effects</expression></expressions>
```
Expected: fire semantic production, optionally reception if acclaim is
explicit; production_techniques carries practical effects.
</example>

<example>
Input:
```xml
<retrieval_intent>Find visually stunning films.</retrieval_intent>
<expressions><expression>visually stunning</expression></expressions>
```
Expected: fire semantic reception/viewer-facing visual praise as supported;
do not invent a named technique.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Roger Deakins movies.</retrieval_intent>
<expressions><expression>Roger Deakins</expression></expressions>
```
Expected: no-fire; literal below-the-line credit, not visual craft acclaim.
</example>

