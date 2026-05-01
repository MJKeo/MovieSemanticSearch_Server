# Few-shot examples

<example>
Input:
```xml
<retrieval_intent>Find films built around a coming-of-age story shape.</retrieval_intent>
<expressions><expression>coming-of-age stories</expression></expressions>
```
Expected: preferred-only when a direct coming-of-age tag exists.
Binary framing makes Keyword appropriate.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with grief present as a light thematic thread.</retrieval_intent>
<expressions><expression>kind of about grief</expression></expressions>
```
Expected: fallback-only. Gradient framing bypasses Keyword even if a
nearby tag exists; Semantic plot_analysis carries degree.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with a redemptive arc as a directional lean.</retrieval_intent>
<expressions><expression>leans redemptive</expression></expressions>
```
Expected: fallback-only. "Leans" is spectrum framing, not binary tag
membership.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is a named historical figure.</retrieval_intent>
<expressions><expression>about JFK</expression></expressions>
```
Expected: no-fire. Concrete subject belongs to Central topic.
</example>
