# Specific praise / criticism - few-shot examples

<example>
Input:
```xml
<retrieval_intent>Find films praised for tension.</retrieval_intent>
<expressions><expression>praised for tension</expression></expressions>
```
Expected: fire semantic on `reception`. Aspect-level praise of the
tension axis; phrase the body as evaluative reception prose, not
mere presence of tension.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films criticized as plodding.</retrieval_intent>
<expressions><expression>criticized as plodding</expression></expressions>
```
Expected: fire semantic on `reception` with criticism language
about pacing. Do not route to runtime metadata or pacing tags —
the user is asking about the *judgment*, not the underlying
attribute.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with broad whole-work cultural reputation.</retrieval_intent>
<expressions><expression>cult classic</expression></expressions>
```
Expected: no-fire. Whole-work cultural status is owned by Cultural
status, not aspect-level praise.
</example>

<example>
Input:
```xml
<retrieval_intent>Find broadly highly-rated movies.</retrieval_intent>
<expressions><expression>highly rated</expression></expressions>
```
Expected: no-fire. Numeric / quality-prior framing is owned by
General appeal, not aspect-level praise.
</example>
