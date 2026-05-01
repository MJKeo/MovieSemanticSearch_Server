# Few-shot examples - Narrative setting

<example>
Input:
```xml
<retrieval_intent>Find films whose story is set in 1940s Berlin.</retrieval_intent>
<expressions><expression>1940s Berlin</expression></expressions>
```
Expected: fire semantic plot_events; plot_summary states a story set in
1940s Berlin without inventing plot.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies that take place on a remote island.</retrieval_intent>
<expressions><expression>remote island</expression></expressions>
```
Expected: fire semantic plot_events; narrative place setting.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies released in the 1940s.</retrieval_intent>
<expressions><expression>1940s</expression></expressions>
```
Expected: no-fire; release date metadata, not story setting.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies filmed in Berlin.</retrieval_intent>
<expressions><expression>Berlin</expression></expressions>
```
Expected: no-fire; filming location, not narrative setting.
</example>

