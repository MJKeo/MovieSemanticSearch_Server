# Few-shot examples - Runtime

<example>
Input:
```xml
<retrieval_intent>Find funny movies under two hours.</retrieval_intent>
<expressions><expression>under 2 hours</expression></expressions>
```
Expected: fire metadata; runtime ceiling at 120 minutes.
</example>

<example>
Input:
```xml
<retrieval_intent>Prefer movies around an hour and a half.</retrieval_intent>
<expressions><expression>around 90 minutes</expression></expressions>
```
Expected: fire metadata; runtime target/range centered near 90 minutes.
</example>

<example>
Input:
```xml
<retrieval_intent>Find long epic-length movies.</retrieval_intent>
<expressions><expression>epic length</expression></expressions>
```
Expected: fire metadata; long-runtime preference, not franchise size.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies in a long-running franchise.</retrieval_intent>
<expressions><expression>long-running</expression></expressions>
```
Expected: no-fire; this is franchise lineage, not runtime.
</example>

