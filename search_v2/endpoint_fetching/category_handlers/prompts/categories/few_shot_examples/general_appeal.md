# Few-shot examples - General appeal / quality baseline

<example>
Input:
```xml
<retrieval_intent>Prefer well-received comedies.</retrieval_intent>
<expressions><expression>well-received</expression></expressions>
```
Expected: fire metadata; reception-quality prior, not a hard threshold.
</example>

<example>
Input:
```xml
<retrieval_intent>Find popular action movies.</retrieval_intent>
<expressions><expression>popular</expression></expressions>
```
Expected: fire metadata; static popularity prior, not live trending.
</example>

<example>
Input:
```xml
<retrieval_intent>Find the best horror movies from the 1980s.</retrieval_intent>
<expressions><expression>best</expression></expressions>
```
Expected: fire metadata for the broad quality prior only; horror and era are
sibling traits.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies trending right now.</retrieval_intent>
<expressions><expression>trending right now</expression></expressions>
```
Expected: no-fire; live buzz belongs to Trending.
</example>

