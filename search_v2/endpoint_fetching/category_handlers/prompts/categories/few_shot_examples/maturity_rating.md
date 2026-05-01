# Few-shot examples - Maturity rating

<example>
Input:
```xml
<retrieval_intent>Find movies rated PG-13 or below.</retrieval_intent>
<expressions><expression>PG-13 or below</expression></expressions>
```
Expected: fire metadata; maturity ceiling at PG-13.
</example>

<example>
Input:
```xml
<retrieval_intent>Find R-rated thrillers.</retrieval_intent>
<expressions><expression>rated R</expression></expressions>
```
Expected: fire metadata; exact R rating.
</example>

<example>
Input:
```xml
<retrieval_intent>Exclude movies rated R.</retrieval_intent>
<expressions><expression>no R-rated</expression></expressions>
```
Expected: fire metadata; target the R rating directly; wrapper polarity
carries exclusion.
</example>

<example>
Input:
```xml
<retrieval_intent>Find family-friendly adventure movies.</retrieval_intent>
<expressions><expression>family-friendly</expression></expressions>
```
Expected: no-fire; audience suitability is not a literal rating.
</example>

