# Few-shot examples

<example>
Input:
```xml
<retrieval_intent>Find films using a nonlinear story structure.</retrieval_intent>
<expressions><expression>nonlinear timeline</expression></expressions>
```
Expected: preferred-only when a direct structure tag exists.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films built around an intimate two-character structure.</retrieval_intent>
<expressions><expression>two-hander</expression></expressions>
```
Expected: fallback-only if no direct structural tag exists. Semantic
narrative_techniques carries the device.
</example>

<example>
Input:
```xml
<retrieval_intent>Find single-location films that withhold key information.</retrieval_intent>
<expressions><expression>single-location with withheld information</expression></expressions>
```
Expected: split if single-location is tagged and the information-control
mechanic is not; do not duplicate the tagged device in Semantic.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with a satisfying happy ending.</retrieval_intent>
<expressions><expression>happy ending</expression></expressions>
```
Expected: no-fire. Ending aftertaste belongs to Emotional / experiential.
</example>
