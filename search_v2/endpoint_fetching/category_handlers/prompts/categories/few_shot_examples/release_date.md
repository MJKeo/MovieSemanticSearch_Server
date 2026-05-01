# Few-shot examples - Release date / era

Use these for shape calibration, not as a closed list.

<example>
Input:
```xml
<retrieval_intent>Find action movies released in the 1990s.</retrieval_intent>
<expressions><expression>90s</expression></expressions>
```
Expected: fire metadata; release-date window for 1990-1999.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies released before 2000.</retrieval_intent>
<expressions><expression>before 2000</expression></expressions>
```
Expected: fire metadata; release-date upper bound before 2000.
</example>

<example>
Input:
```xml
<retrieval_intent>Prefer recent movies.</retrieval_intent>
<expressions><expression>recent</expression></expressions>
```
Expected: fire metadata; vague recent release window, not ordinal.
</example>

<example>
Input:
```xml
<retrieval_intent>Select the newest Scorsese film.</retrieval_intent>
<expressions><expression>newest</expression></expressions>
```
Expected: no-fire; ordinal position belongs to Chronological.
</example>

