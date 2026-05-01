# Few-shot examples - Financial scale

<example>
Input:
```xml
<retrieval_intent>Find big-budget science fiction.</retrieval_intent>
<expressions><expression>big-budget</expression></expressions>
```
Expected: fire metadata; budget scale high.
</example>

<example>
Input:
```xml
<retrieval_intent>Find box-office flops.</retrieval_intent>
<expressions><expression>flop</expression></expressions>
```
Expected: fire metadata; box-office underperformance, not quality.
</example>

<example>
Input:
```xml
<retrieval_intent>Find blockbuster action movies.</retrieval_intent>
<expressions><expression>blockbuster</expression></expressions>
```
Expected: fire metadata; financial compound with large budget and large
gross when both are implied.
</example>

<example>
Input:
```xml
<retrieval_intent>Find cult classics.</retrieval_intent>
<expressions><expression>cult classic</expression></expressions>
```
Expected: no-fire; cultural status, not financial scale.
</example>

