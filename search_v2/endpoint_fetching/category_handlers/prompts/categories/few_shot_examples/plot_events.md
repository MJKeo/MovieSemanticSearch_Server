# Few-shot examples - Plot events

<example>
Input:
```xml
<retrieval_intent>Find films where a heist falls apart after one crew member betrays the others.</retrieval_intent>
<expressions><expression>heist unravels after crew betrayal</expression></expressions>
```
Expected: fire semantic plot_events; dense synopsis-style plot_summary.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films about strangers trapped together who discover one of them is dangerous.</retrieval_intent>
<expressions><expression>strangers trapped together with hidden threat</expression></expressions>
```
Expected: fire semantic plot_events; concrete situation, no theme padding.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films about grief and reconciliation.</retrieval_intent>
<expressions><expression>grief and reconciliation</expression></expressions>
```
Expected: no-fire; thematic archetype/plot analysis, not concrete event.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with a tense, unsettling mood.</retrieval_intent>
<expressions><expression>tense unsettling</expression></expressions>
```
Expected: no-fire; viewer experience, not plot event.
</example>

