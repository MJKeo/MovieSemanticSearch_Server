# Few-shot examples - Chronological ordinal

<example>
Input:
```xml
<retrieval_intent>Select the newest film in the scoped set.</retrieval_intent>
<expressions><expression>newest</expression></expressions>
```
Expected: fire metadata; sort by release date descending and select the top
position; sibling traits define the scoped set.
</example>

<example>
Input:
```xml
<retrieval_intent>Select the earliest Kubrick film.</retrieval_intent>
<expressions><expression>earliest</expression></expressions>
```
Expected: fire metadata; sort by release date ascending and select the top
position; do not emit the director slice here.
</example>

<example>
Input:
```xml
<retrieval_intent>Select the latest Marvel movie.</retrieval_intent>
<expressions><expression>latest</expression></expressions>
```
Expected: fire metadata; latest-release ordinal within the sibling franchise
scope.
</example>

<example>
Input:
```xml
<retrieval_intent>Prefer recent movies.</retrieval_intent>
<expressions><expression>recent</expression></expressions>
```
Expected: no-fire; "recent" is a release-date window, not ordinal.
</example>

