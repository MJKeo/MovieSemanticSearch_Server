# Few-shot examples - Country of origin

<example>
Input:
```xml
<retrieval_intent>Find films produced in France.</retrieval_intent>
<expressions><expression>produced in France</expression></expressions>
```
Expected: fire metadata; production country France.
</example>

<example>
Input:
```xml
<retrieval_intent>Find British productions.</retrieval_intent>
<expressions><expression>British production</expression></expressions>
```
Expected: fire metadata; production country United Kingdom/Britain as the
supported country registry resolves it.
</example>

<example>
Input:
```xml
<retrieval_intent>Find European productions.</retrieval_intent>
<expressions><expression>European productions</expression></expressions>
```
Expected: fire metadata; expand region to supported production countries
only if the endpoint/registry supports that resolution.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies shot in New Zealand.</retrieval_intent>
<expressions><expression>shot in New Zealand</expression></expressions>
```
Expected: no-fire; filming location, not production origin.
</example>

