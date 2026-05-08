# Emotional / experiential - few-shot examples

<example>
Input:
```xml
<retrieval_intent>Find films designed to make the viewer cry.</retrieval_intent>
<expressions><expression>tearjerker films</expression></expressions>
```
Expected: fire semantic across `watch_context` (the viewer's
self-experience goal), `viewer_experience` (cathartic emotional
palette), and `reception` (audience-label "tearjerker"). Felt
effect lives across all three spaces; populating only one
under-covers.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose ending lands as a downer / leaves the viewer subdued.</retrieval_intent>
<expressions><expression>downer ending</expression></expressions>
```
Expected: fire semantic on `viewer_experience.ending_aftertaste`.
The aftertaste vocabulary directly captures the ending shape;
other spaces add noise here.
</example>

<example>
Input:
```xml
<retrieval_intent>Find a movie for Saturday morning kids viewing.</retrieval_intent>
<expressions><expression>Saturday morning kids viewing</expression></expressions>
```
Expected: no-fire. Concrete viewing situation belongs to Viewing
occasion, not emotional/experiential.
</example>

<example>
Input:
```xml
<retrieval_intent>Find horror films.</retrieval_intent>
<expressions><expression>horror</expression></expressions>
```
Expected: no-fire. Bare genre label belongs to Genre, not the
emotional/experiential axis.
</example>
