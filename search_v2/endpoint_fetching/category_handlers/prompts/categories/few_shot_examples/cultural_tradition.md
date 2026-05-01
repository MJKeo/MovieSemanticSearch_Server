# Few-shot examples

<example>
Input:
```xml
<retrieval_intent>Find films in a named cinema tradition.</retrieval_intent>
<expressions><expression>Bollywood</expression></expressions>
```
Expected: preferred-only if the registry directly covers the tradition.
Do not also run country/language metadata.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films in a less-indexed national cinema tradition.</retrieval_intent>
<expressions><expression>Senegalese cinema</expression></expressions>
```
Expected: fallback-only to Metadata country/language if no tradition
tag exists. Name the proxy as lossy.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films in a named movement.</retrieval_intent>
<expressions><expression>French New Wave</expression></expressions>
```
Expected: preferred-only if a movement tag covers it; otherwise fallback
to the best country/language proxy without inventing movement tags.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose legal production country is France.</retrieval_intent>
<expressions><expression>made in France</expression></expressions>
```
Expected: no-fire. Production-country wording belongs to Country of
origin.
</example>
