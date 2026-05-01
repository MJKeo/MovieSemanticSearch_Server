# Few-shot examples - Numeric reception score

<example>
Input:
```xml
<retrieval_intent>Find movies rated above 8.</retrieval_intent>
<expressions><expression>rated above 8</expression></expressions>
```
Expected: fire metadata; numeric reception threshold greater than 8.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies with at least 70 percent approval.</retrieval_intent>
<expressions><expression>70%+</expression></expressions>
```
Expected: fire metadata; numeric/percent reception threshold.
</example>

<example>
Input:
```xml
<retrieval_intent>Find five-star movies.</retrieval_intent>
<expressions><expression>5-star</expression></expressions>
```
Expected: fire metadata; star-scale reception value if supported by the
endpoint's numeric surface.
</example>

<example>
Input:
```xml
<retrieval_intent>Find highly rated thrillers.</retrieval_intent>
<expressions><expression>highly rated</expression></expressions>
```
Expected: no-fire; qualitative quality prior belongs to General appeal.
</example>

