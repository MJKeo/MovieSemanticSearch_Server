# Few-shot examples - Streaming platform

<example>
Input:
```xml
<retrieval_intent>Find sci-fi thrillers available on Netflix.</retrieval_intent>
<expressions><expression>on Netflix</expression></expressions>
```
Expected: fire metadata; provider availability on Netflix.
</example>

<example>
Input:
```xml
<retrieval_intent>Find comedies streaming on Hulu.</retrieval_intent>
<expressions><expression>streaming on Hulu</expression></expressions>
```
Expected: fire metadata; provider availability on Hulu.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies available to rent.</retrieval_intent>
<expressions><expression>available to rent</expression></expressions>
```
Expected: fire metadata only if the endpoint supports access mode; do not
invent a provider.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Disney animated movies.</retrieval_intent>
<expressions><expression>Disney</expression></expressions>
```
Expected: no-fire; Disney is studio/brand here, not streaming availability.
</example>

