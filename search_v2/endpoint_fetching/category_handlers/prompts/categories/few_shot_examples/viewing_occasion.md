# Few-shot examples - Viewing occasion

<example>
Input:
```xml
<retrieval_intent>Find movies suitable for date night.</retrieval_intent>
<expressions><expression>date night</expression></expressions>
```
Expected: fire semantic watch_context; `watch_scenarios`: date night.
</example>

<example>
Input:
```xml
<retrieval_intent>Find something to watch in the background while cooking.</retrieval_intent>
<expressions><expression>background watching while cooking</expression></expressions>
```
Expected: fire semantic watch_context; practical viewing scenario.
</example>

<example>
Input:
```xml
<retrieval_intent>Find a movie for a long flight.</retrieval_intent>
<expressions><expression>long flight</expression></expressions>
```
Expected: fire semantic watch_context; occasion, not plot or tone.
</example>

<example>
Input:
```xml
<retrieval_intent>Find something uplifting.</retrieval_intent>
<expressions><expression>uplifting</expression></expressions>
```
Expected: no-fire; emotional experience, not a concrete viewing occasion.
</example>

