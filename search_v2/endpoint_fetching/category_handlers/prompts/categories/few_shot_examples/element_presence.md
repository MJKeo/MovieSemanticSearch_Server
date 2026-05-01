# Few-shot examples

<example>
Input:
```xml
<retrieval_intent>Find films where zombies are present as the story element.</retrieval_intent>
<expressions><expression>zombie movies</expression></expressions>
```
Expected: preferred-only when the registry has a direct zombie tag.
Semantic would duplicate presence and add noise.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films featuring a long-tail concrete animal element.</retrieval_intent>
<expressions><expression>movies with horses</expression></expressions>
```
Expected: fallback-only if no direct horse tag exists. Semantic
plot_events carries concrete presence.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with subtle vampire motifs rather than literal vampire membership.</retrieval_intent>
<expressions><expression>subtle vampire motifs</expression></expressions>
```
Expected: fallback-only or split only if a direct tag covers literal
presence and Semantic is needed for subtle / motif-level qualification.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose story is about an underdog arc.</retrieval_intent>
<expressions><expression>underdog stories</expression></expressions>
```
Expected: no-fire. This is story shape, not element presence.
</example>
