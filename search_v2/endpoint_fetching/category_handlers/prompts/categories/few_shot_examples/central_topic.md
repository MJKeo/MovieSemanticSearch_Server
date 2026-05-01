# Few-shot examples

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is a named public figure.</retrieval_intent>
<expressions><expression>about Princess Diana</expression></expressions>
```
Expected: fallback-only. Preferred Keyword has no Diana-specific tag;
Semantic plot_events carries the named subject. Do not use a generic
biography tag as the answer.
</example>

<example>
Input:
```xml
<retrieval_intent>Find biographical films as a subject class.</retrieval_intent>
<expressions><expression>biopics</expression></expressions>
```
Expected: preferred-only if the registry directly covers biographical
subject films. No Semantic gap-fill needed for a bare class request.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films about a specific political scandal.</retrieval_intent>
<expressions><expression>about Watergate</expression></expressions>
```
Expected: fallback-only unless a direct Watergate tag exists. Semantic
plot_events should name the concrete subject, not invent plot details.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with grief as the core theme.</retrieval_intent>
<expressions><expression>about grief</expression></expressions>
```
Expected: no-fire. This is thematic essence, not a concrete focal
subject.
</example>
