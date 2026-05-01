# Few-shot examples - Filming location

<example>
Input:
```xml
<retrieval_intent>Find movies physically filmed in New Zealand.</retrieval_intent>
<expressions><expression>New Zealand</expression></expressions>
```
Expected: fire semantic production; `filming_locations`: New Zealand.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films shot on location in Iceland.</retrieval_intent>
<expressions><expression>Iceland</expression></expressions>
```
Expected: fire semantic production; compact place name only.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose story takes place in Tokyo.</retrieval_intent>
<expressions><expression>Tokyo</expression></expressions>
```
Expected: no-fire; narrative setting, not filming location.
</example>

<example>
Input:
```xml
<retrieval_intent>Find French-produced films.</retrieval_intent>
<expressions><expression>France</expression></expressions>
```
Expected: no-fire; production country metadata, not shooting geography.
</example>

