# Few-shot examples - Studio / brand

<example>
Input:
```xml
<retrieval_intent>Find horror films produced by A24.</retrieval_intent>
<expressions><expression>A24</expression></expressions>
```
Expected: fire studio endpoint; curated brand path for A24.
</example>

<example>
Input:
```xml
<retrieval_intent>Find classic MGM musicals; the era is handled by a separate release-era trait.</retrieval_intent>
<expressions><expression>MGM</expression></expressions>
```
Expected: fire studio endpoint; MGM brand path; do not encode the era here.
</example>

<example>
Input:
```xml
<retrieval_intent>Find action movies produced by Cannon Films.</retrieval_intent>
<expressions><expression>Cannon Films</expression></expressions>
```
Expected: fire studio endpoint; freeform credited names for the long-tail
studio.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies available on Netflix.</retrieval_intent>
<expressions><expression>Netflix</expression></expressions>
```
Expected: no-fire; streaming availability is metadata, not production brand.
</example>

