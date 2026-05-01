# Few-shot examples - Dialogue craft acclaim

<example>
Input:
```xml
<retrieval_intent>Find films praised for quotable dialogue.</retrieval_intent>
<expressions><expression>quotable dialogue</expression></expressions>
```
Expected: fire semantic reception; praised_qualities targets dialogue.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies with snappy rapid-fire banter.</retrieval_intent>
<expressions><expression>snappy rapid-fire banter</expression></expressions>
```
Expected: fire semantic narrative_techniques, optionally reception if praise
is explicit.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with naturalistic overlapping dialogue.</retrieval_intent>
<expressions><expression>naturalistic overlapping dialogue</expression></expressions>
```
Expected: fire semantic narrative_techniques; dialogue craft pattern.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies written by Aaron Sorkin.</retrieval_intent>
<expressions><expression>Aaron Sorkin</expression></expressions>
```
Expected: no-fire; writer credit belongs to person credit.
</example>

